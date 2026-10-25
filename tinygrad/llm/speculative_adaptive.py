"""Per-request fallback from measured, complete speculative round costs."""
import math


class DecodeFallbackPolicy:
  def __init__(self, reference_seconds_per_token:float, min_blocks:int=2, max_slowdown:float=1.1,
               max_extra_seconds:float|None=None):
    if (type(reference_seconds_per_token) not in (int,float) or not math.isfinite(reference_seconds_per_token)
        or reference_seconds_per_token<=0):raise ValueError('Ordinary decode reference must be finite and positive')
    if type(min_blocks) is not int or min_blocks<1:raise ValueError('Minimum block count must be a positive integer')
    if type(max_slowdown) not in (int,float) or not math.isfinite(max_slowdown) or max_slowdown<1:
      raise ValueError('Maximum slowdown must be finite and at least one')
    if max_extra_seconds is not None and (type(max_extra_seconds) not in (int,float)
        or not math.isfinite(max_extra_seconds) or max_extra_seconds<=0):
      raise ValueError('Maximum extra compute time must be finite and positive')
    self.reference_seconds_per_token = float(reference_seconds_per_token)
    self.min_blocks,self.max_slowdown = min_blocks,float(max_slowdown)
    self.max_extra_seconds = max_extra_seconds
    self.blocks,self.tokens,self.seconds,self.fallback = 0,0,0.0,False
    self.reason = ''

  def record(self, tokens:int, seconds:float) -> bool:
    """Charge synchronized compute/copy/rollback/observation work, excluding consumer waits."""
    if type(tokens) is not int or tokens<1:raise ValueError('Committed token count must be a positive integer')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
      raise ValueError('Round cost must be finite and positive')
    elapsed = self.seconds+seconds
    if not math.isfinite(elapsed):raise ValueError('Accumulated round cost must be finite')
    self.blocks,self.tokens,self.seconds = self.blocks+1,self.tokens+tokens,elapsed
    # A clearly expensive first block need not consume the minimum sample
    # count. Compare complete compute cost against the same number of ordinary
    # tokens; earlier savings offset later costs. Consumer time is excluded.
    if not self.fallback:
      if self.max_extra_seconds is not None and self.seconds-self.tokens*self.reference_seconds_per_token>self.max_extra_seconds:
        self.fallback,self.reason = True,'measured_extra_cost'
      elif self.blocks>=self.min_blocks and self.seconds/self.tokens>self.reference_seconds_per_token*self.max_slowdown:
        self.fallback,self.reason = True,'measured_round_cost'
    return self.fallback
