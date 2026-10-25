"""Opt-in speculative budgets selected from complete, qualified decode-round timings.

This independent throughput policy is not an EAGLE-4 adapter. It neither predicts correctness nor implements drafting,
tree verification or cache rollback. Callers supply those operations and time the complete round, including rejected
proposal replay, state copies, synchronization and observation. Warmup and unqualified runs cannot enable a budget.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PolicyKey:
  target_revision: str
  draft_revision: str
  runtime: str  # Device, kernels, backend options, thread count and driver/build identity.
  precision: str
  batch_size: int
  context_range: tuple[int, int]  # Inclusive measured band; never extrapolated to another band.

  def __post_init__(self):
    if any(not isinstance(value, str) or not value for value in
           (self.target_revision, self.draft_revision, self.runtime, self.precision)):
      raise ValueError("model revisions, runtime and precision must be nonempty strings")
    if type(self.batch_size) is not int or self.batch_size < 1:raise ValueError("batch_size must be a positive integer")
    if (type(self.context_range) is not tuple or len(self.context_range) != 2 or
        any(type(value) is not int for value in self.context_range) or
        not 0 <= self.context_range[0] <= self.context_range[1]):raise ValueError("invalid measured context band")


@dataclass(frozen=True)
class PolicyDecision:
  budget: int  # Proposed tokens, excluding the ordinary target/bonus token. Zero means ordinary decoding.
  speedup: float
  samples: int
  reason: str


class MeasuredSpeculativePolicy:
  def __init__(self, budgets: tuple[int, ...], *, min_samples: int = 3, min_speedup: float = 1.05,
               window: int = 32, max_profiles: int = 16):
    if (not budgets or any(type(value) is not int or value <= 0 for value in budgets) or len(set(budgets)) != len(budgets)):
      raise ValueError("budgets must be distinct positive integers supported by the drafter")
    if any(type(value) is not int or value < 1 for value in (min_samples, window, max_profiles)) or min_samples > window:
      raise ValueError("positive sample/profile limits and min_samples <= window required")
    if not math.isfinite(min_speedup) or min_speedup < 1:raise ValueError("min_speedup must be finite and >= 1")
    self.budgets, self.min_samples, self.min_speedup = tuple(sorted(budgets)), min_samples, min_speedup
    self.window, self.max_profiles = window, max_profiles
    self._profiles: OrderedDict[PolicyKey, dict[int, deque[tuple[int, float]]]] = OrderedDict()

  def record(self, key: PolicyKey, budget: int, emitted_tokens: int, total_seconds: float, *,
             qualified: bool = False, warmup: bool = False) -> None:
    """Record one successful round. ``qualified`` requires external output/state correctness qualification.

    ``total_seconds`` must include every phase through committed output, not just draft and target forward times.
    A caller measuring an asynchronous backend must synchronize before ending the timer. This is per-request
    latency; emitted_tokens is not multiplied by batch size. Use separate keys for distinct serving loads.
    """
    if type(budget) is not int or budget not in (0, *self.budgets):raise ValueError("unsupported proposal budget")
    if type(emitted_tokens) is not int or not 1 <= emitted_tokens <= budget + 1:raise ValueError("invalid emitted token count")
    if not math.isfinite(total_seconds) or total_seconds <= 0:raise ValueError("round timing must be finite and positive")
    if not qualified or warmup:return
    if key not in self._profiles:
      if len(self._profiles) >= self.max_profiles:self._profiles.popitem(last=False)
      self._profiles[key] = {}
    self._profiles.move_to_end(key)
    self._profiles[key].setdefault(budget, deque(maxlen=self.window)).append((emitted_tokens, total_seconds))

  def invalidate(self, key: PolicyKey, budget: int) -> None:
    """Drop a budget's evidence after a failed correctness check or changed calibration contract."""
    if key in self._profiles:self._profiles[key].pop(budget, None)

  def choose(self, key: PolicyKey, *, context_tokens: int, remaining_tokens: int) -> PolicyDecision:
    """Select a measured budget. ``remaining_tokens`` is capped by both request output and model context room."""
    if type(context_tokens) is not int or context_tokens < 0:raise ValueError("invalid context length")
    if type(remaining_tokens) is not int or remaining_tokens < 0:raise ValueError("invalid remaining token count")
    if remaining_tokens <= 1:return PolicyDecision(0, 1.0, 0, "insufficient_output_budget")
    if not key.context_range[0] <= context_tokens <= key.context_range[1]:
      return PolicyDecision(0, 1.0, 0, "outside_measured_context")
    if key not in self._profiles:return PolicyDecision(0, 1.0, 0, "no_matching_measurements")
    self._profiles.move_to_end(key)
    samples = self._profiles[key]
    baseline = samples.get(0, ())
    if len(baseline) < self.min_samples:return PolicyDecision(0, 1.0, len(baseline), "insufficient_ordinary_samples")
    ordinary_rate = sum(row[0] for row in baseline) / sum(row[1] for row in baseline)
    best = PolicyDecision(0, 1.0, len(baseline), "no_measured_gain")
    for budget in self.budgets:
      if budget + 1 > remaining_tokens:continue
      candidate = samples.get(budget, ())
      if len(candidate) < self.min_samples:continue
      speedup = (sum(row[0] for row in candidate) / sum(row[1] for row in candidate)) / ordinary_rate
      if speedup > max(self.min_speedup, best.speedup):best = PolicyDecision(budget, speedup, len(candidate), "measured_gain")
    return best
