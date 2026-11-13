#!/usr/bin/env python3
"""Count iommu-domain switches in the recorded decode submit order (no NPU needed).

Reads an ONPU_REPLAY_PROFILE jsonl (one line per decoded token) and extracts the
NPU device domain of every NPU call in execution order:
    'ROCKCHIP:<kernel>'   -> domain 0
    'ROCKCHIP:1:<kernel>' -> domain 1
    'ROCKCHIP:2:<kernel>' -> domain 2
Consecutive calls with the same domain are one run; a switch happens between runs.
The number of runs is the number of blocking submits that end up paying the driver's
iommu detach/attach (rknpu_iommu_domain_get_and_switch) on the shipped tree.

Also prints the per-layer pattern (the first N domains) and the run-length histogram,
and can compare two profiles (`--compare baseline.jsonl --compare other.jsonl`).

  python3 /home/orangepi/tg-exp-stream/scripts/analyze_submit_domains.py \
      --profile /home/orangepi/oNPU/analysis/arms_cbuf/20261109-202029/r1-narrow_cbuf/narrow_cbuf.profile.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path


def domain_of(label: str) -> int | None:
  if not label.startswith("ROCKCHIP"): return None
  parts = label.split(":")
  if len(parts) == 2: return 0
  try: return int(parts[1])
  except ValueError: return 0


# ChatGLM3-6B decode: the six NPU calls of one layer, in order (name suffixes), then lm_head.
LAYER_PATTERN = ("rockchip_int8_1_4608_4096", "rockchip_int8_1_4096_4096", "rockchip_int8_1_24576_4096",
                 "rockchip_int8_1_2816_4096", "rockchip_grouped_int8_3_1_4096_4096_3",
                 "rockchip_int8_1_4096_1536")
LMHEAD_PATTERN = ("rockchip_int8_1_15872_4096", "rockchip_int8_1_24576_4096", "rockchip_int8_1_24576_4096")


def layer_segments(domains: list[int], names: list[str]) -> list[list[int]]:
  """Split the NPU call order into layers by the known ChatGLM3 per-layer kernel pattern."""
  segments, index = [], 0
  while index + len(LAYER_PATTERN) <= len(names) and tuple(names[index:index + len(LAYER_PATTERN)]) == LAYER_PATTERN:
    segments.append(domains[index:index + len(LAYER_PATTERN)])
    index += len(LAYER_PATTERN)
  if tuple(names[index:index + len(LMHEAD_PATTERN)]) == LMHEAD_PATTERN:
    segments.append(domains[index:index + len(LMHEAD_PATTERN)])
  return segments


def analyse(profile: Path) -> dict:
  first = json.loads(profile.read_text().splitlines()[0])
  labels = first["labels"]
  npu_labels = [label for label in labels if domain_of(label) is not None]
  domains = [domain_of(label) for label in npu_labels]
  names = [label.split(":", 1)[1] if domain_of(label) == 0 else label.split(":", 2)[2] for label in npu_labels]
  runs: list[int] = []
  for domain in domains:
    if not runs or runs[-1] != domain: runs.append(domain)
  switches = max(0, len(runs) - 1)
  run_lengths = [len(list(group)) for _, group in itertools.groupby(domains)]
  segments = layer_segments(domains, names)
  uniform = sum(1 for segment in segments if len(set(segment)) == 1)
  result = {"profile": str(profile), "npu_calls_per_token": len(domains), "domains": domains,
            "domain_sequence": runs, "runs_per_token": len(runs), "switches_per_token": switches,
            "run_length_histogram": dict(sorted(Counter(run_lengths).items())),
            "per_layer_calls_head": domains[:6 * 6],
            "layer_segments": len(segments), "uniform_layers": uniform,
            "block_major": bool(segments) and uniform == len(segments),
            "layer_domains": [segment[0] for segment in segments],
            "domain_changes_between_layers": sum(1 for a, b in itertools.pairwise(segments) if a[0] != b[0]),
            "submit_profile_submits": first.get("submits"), "submit_profile_tasks": first.get("tasks")}
  means = first.get("ns", {})
  if means: result["mean_us_per_label"] = {label: round(means[label] / 1000 / 1, 1) for label in npu_labels[:3]}
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile", required=True)
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()
  result = analyse(Path(args.profile))
  # compact print: full domain sequence only for the first 120 entries
  result["domain_sequence_head"] = result.pop("domain_sequence")[:120]
  print(json.dumps(result, indent=1))
  if args.json_out:
    with open(args.json_out, "w") as f: f.write(json.dumps(result, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
