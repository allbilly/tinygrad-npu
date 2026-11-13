#!/usr/bin/env python3
"""Re-derive the NPU per-kernel model and submit accounting straight from the
ONPU_REPLAY_PROFILE artifacts (no prose, no handoff numbers).

Each *.profile.jsonl line is one NativeReplay call (one decoded token) and the
`ns`/`counts` dicts are CUMULATIVE for that NativeReplay instance; the prefill
replay is interleaved as one line with submits=479. A per-token delta is only
valid between two decode lines (submits==171).

What this prints:
  1. per-label calls/token, ms/token, us/call, MB/call (INT8 weights), GB/s,
     tasks/call
  2. the submit fixed-cost fit ms_per_submit = a + b*MB_per_submit, with R^2 and
     residual table; also the a=5.25us alternative's systematic residual
  3. the byte/submit accounting for every projection (where the 171 submits are)
  4. per-token NPU-sum distribution (drift / clock-bucket evidence inside one run)
  5. what hypothetical merges would save, in ms, using the fitted a

Usage:
  python3 analyze_profile_stream.py --root /home/orangepi/oNPU/analysis/arms_cbuf/20261109-202029 \
      --arm narrow_cbuf --arm base
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# INT8 single-row matmul: weight bytes = n*k (1 byte/weight).
# Names are emitted as rockchip_int8_{m}_{n}_{k} / rockchip_grouped_int8_{groups}_{m}_{n}_{k}_{tile}.
WEIGHT_BYTES = {
  "rockchip_int8_1_4608_4096": 4608 * 4096,
  "rockchip_int8_1_4096_4096": 4096 * 4096,
  "rockchip_int8_1_24576_4096": 24576 * 4096,
  "rockchip_int8_1_2816_4096": 2816 * 4096,
  "rockchip_int8_1_4096_1536": 4096 * 1536,
  "rockchip_int8_1_15872_4096": 15872 * 4096,
  "rockchip_grouped_int8_3_1_4096_4096_3": 3 * 4096 * 4096,
}
# Which projection each kernel is (for the accounting table).
ROLE = {
  "rockchip_int8_1_4608_4096": "qkv",
  "rockchip_int8_1_4096_4096": "attn_out",
  "rockchip_int8_1_24576_4096": "gate+up main (or lm_head)",
  "rockchip_int8_1_2816_4096": "gate+up tail",
  "rockchip_int8_1_4096_1536": "ffn_down k-tail (1408 padded to 1536)",
  "rockchip_grouped_int8_3_1_4096_4096_3": "ffn_down main (3 k-parts, 1 grouped call)",
  "rockchip_int8_1_15872_4096": "lm_head first slice",
}


def kernel_of(label: str) -> str | None:
  # 'ROCKCHIP' / 'ROCKCHIP:1' / 'ROCKCHIP:2' device prefix
  parts = label.split(":")
  if parts[0] != "ROCKCHIP" or len(parts) < 2:
    return None
  return parts[-1]  # strip the optional device-domain suffix ROCKCHIP:1:<kernel>


def load_deltas(path: Path):
  lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
  keys = set()
  for d in lines:
    keys |= set(d["ns"])
  deltas = []  # list[dict[label, ns]] one per decoded token
  for i in range(1, len(lines)):
    a, b = lines[i - 1], lines[i]
    # A prefill replay line resets/interleaves counters; only pure decode->decode deltas.
    if a["submits"] != 171 or b["submits"] != 171:
      continue
    if any(b["counts"].get(k, 0) < a["counts"].get(k, 0) for k in b["counts"]):
      continue
    deltas.append({k: b["ns"].get(k, 0) - a["ns"].get(k, 0) for k in keys})
  return deltas


def fit(rows, fixed_us_override=None):
  """rows: list[(mb_per_submit, ms_per_submit)]. Returns a_us, b_ms_per_MB, r2, resid."""
  xs = [r[0] for r in rows]
  ys = [r[1] for r in rows]
  n = len(xs)
  mx, my = statistics.mean(xs), statistics.mean(ys)
  sxx = sum((x - mx) ** 2 for x in xs)
  sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
  b = sxy / sxx
  a = my - b * mx
  if fixed_us_override is not None:
    a = fixed_us_override / 1000.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) - a * 0  # not used
    b = (sum(y - a for y in ys)) / sum(xs)  # least-squares through fixed a
    b = (sum(x * (y - a) for x, y in zip(xs, ys))) / sum(x * x for x in xs)
  pred = [a + b * x for x in xs]
  ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
  ss_tot = sum((y - my) ** 2 for y in ys)
  r2 = 1 - ss_res / ss_tot
  resid = [(rows[i][0], rows[i][1], pred[i], (rows[i][1] - pred[i]) / pred[i] * 100, rows[i][2]) for i in range(n)]
  return a * 1000.0, b, r2, resid, sum(pred)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="/home/orangepi/oNPU/analysis/arms_cbuf/20261109-202029")
  ap.add_argument("--arm", action="append", default=[])
  args = ap.parse_args()
  root = Path(args.root)
  arms = args.arm or ["base", "narrow_cbuf"]
  out = {"root": str(root), "arms": {}}
  for arm in arms:
    files = sorted(root.glob(f"r*-{arm}/{arm}.profile.jsonl"))
    deltas = []
    for f in files:
      deltas += load_deltas(f)
    if not deltas:
      continue
    keys = sorted(deltas[0])
    nt = len(deltas)
    # ---- per-label table -------------------------------------------------
    labels = {}
    for k in keys:
      # counts/token for every label comes from the profile's own counts dict below
      vals = [d[k] for d in deltas]
      ms = statistics.mean(vals) / 1e6
      labels[k] = {"ms_per_token": round(ms, 5), "sd_ms": round(statistics.pstdev(vals) / 1e6, 5)}
    # counts/token from first profile line counts (per token constant)
    line0 = json.loads(files[0].read_text().splitlines()[0])
    counts = dict(line0["counts"])
    submits_per_token = line0["submits"]
    tasks_per_token = line0["tasks"]
    for k, v in labels.items():
      v["calls_per_token"] = counts.get(k, 0)
      v["us_per_call"] = round(v["ms_per_token"] * 1000 / max(1, v["calls_per_token"]), 3)
      kern = kernel_of(k)
      if kern in WEIGHT_BYTES:
        mb = WEIGHT_BYTES[kern] / 1e6
        v["mb_per_call"] = round(mb, 3)
        v["gbps"] = round(mb / (v["ms_per_token"] / v["calls_per_token"]), 3) if v["ms_per_token"] else 0
        v["role"] = ROLE[kern]
    labels = {k: v for k, v in labels.items() if v["calls_per_token"] > 0}
    # ROCKCHIP total
    npu_ms = sum(v["ms_per_token"] for k, v in labels.items() if k.startswith("ROCKCHIP"))
    cpu_ms = sum(v["ms_per_token"] for k, v in labels.items() if k.startswith("CPU"))
    copy_ms = sum(v["ms_per_token"] for k, v in labels.items() if k.startswith("COPY"))
    # ---- fit -------------------------------------------------------------
    rows = []
    for k, v in labels.items():
      if "mb_per_call" in v:
        rows.append((v["mb_per_call"], v["ms_per_token"] / v["calls_per_token"], k))
    a_us, b, r2, resid, pred_sum = fit(rows)
    pred_weighted = sum(v["calls_per_token"] * (a_us / 1000.0 + b * v["mb_per_call"])
                        for v in labels.values() if "mb_per_call" in v)
    a_fix_us, b_fix, r2_fix, resid_fix, pred_fix = fit(rows, fixed_us_override=5.25)
    # ---- per-token NPU sum ----------------------------------------------
    tok_sum = [sum(d[k] for k in keys if k.startswith("ROCKCHIP")) / 1e6 for d in deltas]
    tok_tot = [sum(d[k] for k in keys) / 1e6 for d in deltas]
    out["arms"][arm] = {
      "files": [str(f) for f in files],
      "tokens": nt,
      "submits_per_token": submits_per_token,
      "tasks_per_token": tasks_per_token,
      "tasks_per_submit": round(tasks_per_token / submits_per_token, 4),
      "labels": labels,
      "sums_ms": {"rockchip": round(npu_ms, 4), "cpu": round(cpu_ms, 4), "copy": round(copy_ms, 4),
                  "all": round(npu_ms + cpu_ms + copy_ms, 4)},
      "fit": {"a_us_per_submit": round(a_us, 4), "b_ms_per_MB": round(b, 6),
              "asym_gbps": round(1e3 / b / 1000, 3) if b else None, "r2": round(r2, 6),
              "total_ms_predicted": round(pred_sum, 4),
              "total_ms_predicted_weighted": round(pred_weighted, 4),
              "resid_pct": [[r[4], r[0], round(r[1], 5), round(r[2], 5), round(r[3], 2)] for r in resid]},
      "fit_5.25us_alt": {"a_us_per_submit": a_fix_us, "b_ms_per_MB": round(b_fix, 6),
                         "asym_gbps": round(1e3 / b_fix / 1000, 3) if b_fix else None, "r2": round(r2_fix, 6),
                         "total_ms_predicted": round(pred_fix, 4),
                         "resid_pct": [[r[4], r[0], round(r[1], 5), round(r[2], 5), round(r[3], 2)] for r in resid_fix]},
      "per_token_rockchip_sum_ms": {"mean": round(statistics.mean(tok_sum), 4),
                                    "sd": round(statistics.pstdev(tok_sum), 4),
                                    "min": round(min(tok_sum), 4), "max": round(max(tok_sum), 4),
                                    "p10": round(sorted(tok_sum)[nt // 10], 4),
                                    "p90": round(sorted(tok_sum)[9 * nt // 10], 4)},
      "per_token_all_sum_ms": {"mean": round(statistics.mean(tok_tot), 4),
                               "sd": round(statistics.pstdev(tok_tot), 4)},
      "chained_merge_savings": {
        "nparts_3to1_171to57_submits": round((submits_per_token - submits_per_token / 3) * a_us / 1000, 4),
        "tail_2816_merge_28_submits": round(28 * a_us / 1000, 4),
        "lmhead_merge": round(2 * a_us / 1000, 4),
      },
    }
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
