"""CPU-only numeric report for the FP16 NPU MLA prefill attention path (no NPU touched).

Simulates the exact numerics of the wired graph path:
  expansions fp16 x fp16 -> fp32 accumulate; row-major FP16 lhs; QK/PV fp16 x fp16 -> fp32
  accumulate; exact-order CPU softmax on the f32 scores; fp16 probability operand for PV.
against the shipped absorbed FP32 CPU reference (current fused kernel semantics: f32 projections,
f32 qk/pv, exact-order softmax), at the real MiniCPM3-4B shapes but with synthetic activations
(GGUF weights unavailable offline). Reports max/relative error and the downstream dynamic-INT8
code churn of the attention output (the quantity that perturbs attn_output/logits).

Run: python3 scripts/mla_npu_attn_numeric_report.py [--seed N]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HEADS, ROWS, CTX, LATENT, NOPE, ROPE, VDIM = 40, 128, 128, 256, 64, 32, 64
SCALE = np.float32(1.0 / np.sqrt(96.0))
LOG2E = np.float32(1.4426950216293335)
POLY = [np.float32(x) for x in (0.00015359208919107914, 0.0013392627006396651, 0.009618384763598442,
                                0.055503472685813904, 0.24022644758224487, 0.6931471824645996, 1.0)]


def f16(x): return np.asarray(x, dtype=np.float32).astype(np.float16)
def f32(x): return np.asarray(x, dtype=np.float32)


def exp2_poly(x):
  x = f32(x)
  q = np.trunc(x + np.where(x < 0, np.float32(-0.5), np.float32(0.5))).astype(np.int32)
  s = f32(x - q.astype(np.float32))
  y = POLY[0]
  for c in POLY[1:]: y = f32(y * s + c)
  return (y.view(np.int32) + q * np.int32(0x800000)).view(np.float32)


def exact_softmax(scores):
  """Shipped fused-kernel order: causal end=r+1, -inf tail, sequential max, (s-max)*log2e, exp2 poly, sequential sum, divide."""
  probs = np.zeros((HEADS, ROWS, CTX), dtype=np.float32)
  for h in range(HEADS):
    for r in range(ROWS):
      end = r + 1
      sc = f32(f32(scores[h, r, :end]) * SCALE)
      maximum = np.float32(np.max(sc))
      e = exp2_poly(f32(f32(sc - maximum) * LOG2E))
      total = np.float32(0.0)
      t = 0
      while t + 4 <= end:
        for j in range(4): total = np.float32(total + e[t + j])
        t += 4
      while t < end:
        total = np.float32(total + e[t])
        t += 1
      probs[h, r, :end] = f32(e / total)
  return probs


def reference(c_kv, k_rope, q_nope, q_rope, w_kb, w_vb):
  """Shipped absorbed FP32 CPU path: projections, f32 qk(+scale), exact softmax, f32 pv, value projection."""
  q_abs = np.einsum('rhc,hdc->rhd', f32(q_nope), f32(w_kb), optimize=True)          # (rows,heads,latent)
  scores = np.einsum('rhd,td->hrt', q_abs, f32(c_kv), optimize=True)
  scores += np.einsum('rhc,tc->hrt', f32(q_rope), f32(k_rope), optimize=True)
  probs = exact_softmax(scores)
  ctx = np.einsum('hrt,tv->hrv', probs, f32(c_kv), optimize=True)                   # (heads,rows,latent)
  out = np.einsum('hrv,hcv->hrc', ctx, f32(w_vb), optimize=True)                    # value projection
  return out.transpose(1, 0, 2).reshape(ROWS, HEADS * VDIM)


def npu_path(c_kv, k_rope, q_nope, q_rope, w_kb, w_vb):
  """Wired graph numerics: FP16 operands, FP32 accumulation; exact-order softmax on f32 scores; FP16 P."""
  c16, kr16 = f16(c_kv), f16(k_rope)
  q16 = f16(np.concatenate([q_nope, q_rope], axis=-1))                              # (rows,heads,96)
  wkb16, wvb16 = f16(w_kb), f16(w_vb)
  k_nope = np.einsum('tl,hld->htd', c16.astype(np.float32), wkb16.astype(np.float32), optimize=True)
  v_exp = np.einsum('tl,hcl->htc', c16.astype(np.float32), wvb16.astype(np.float32), optimize=True)
  k_full = f16(np.concatenate([k_nope, np.broadcast_to(kr16[None], (HEADS, CTX, ROPE))], axis=-1))
  v_full = f16(v_exp)
  scores = np.einsum('rhq,hkq->hrk', q16.astype(np.float32), k_full.astype(np.float32), optimize=True)
  probs = exact_softmax(scores)
  p16 = f16(probs)
  ctx = np.einsum('hrk,hkv->hrv', p16.astype(np.float32), v_full.astype(np.float32), optimize=True)
  return ctx.transpose(1, 0, 2).reshape(ROWS, HEADS * VDIM)


def quant_codes(x):
  out = np.zeros(x.shape, dtype=np.int32)
  for r in range(x.shape[0]):
    row = f32(x[r])
    largest = np.float32(np.max(np.abs(row)))
    scale = np.float32(largest / np.float32(127.0))
    if scale < np.float32(1e-8): scale = np.float32(1e-8)
    out[r] = np.clip(np.rint(f32(row / scale)).astype(np.int32), -127, 127)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seed', type=int, default=20261110)
  ap.add_argument('--json', type=Path, default=Path(__file__).resolve().parent.parent / 'scripts' / 'mla_npu_attn_numeric_report.json')
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  reports = []
  for wscale in (0.25, 0.0625):   # brackets plausible GGUF weight scales for the absorbed matrices
    c_kv = f32(rng.normal(0, 1.0, (CTX, LATENT)))                 # RMSNorm output, unit RMS
    k_rope = f32(rng.normal(0, 1.0, (CTX, ROPE)))
    q_nope = f32(rng.normal(0, 1.0, (ROWS, HEADS, NOPE)))
    q_rope = f32(rng.normal(0, 1.0, (ROWS, HEADS, ROPE)))
    w_kb = f32(rng.normal(0, wscale, (HEADS, LATENT, NOPE)))
    w_vb = f32(rng.normal(0, wscale, (HEADS, VDIM, LATENT)))
    ref = reference(c_kv, k_rope, q_nope, q_rope, w_kb, w_vb)
    new = npu_path(c_kv, k_rope, q_nope, q_rope, w_kb, w_vb)
    diff = f32(new - ref)
    denom = np.maximum(np.abs(ref), np.float32(1e-6))
    qref, qnew = quant_codes(ref), quant_codes(new)
    row = {
      'weight_scale': wscale,
      'ref_absmax': float(np.max(np.abs(ref))),
      'max_abs_error': float(np.max(np.abs(diff))),
      'max_rel_error': float(np.max(np.abs(diff) / denom)),
      'rel_rms': float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(ref ** 2))),
      'int8_codes': int(qref.size),
      'int8_codes_changed': int(np.count_nonzero(qref != qnew)),
      'int8_codes_changed_pct': 100.0 * float(np.count_nonzero(qref != qnew)) / qref.size,
      'int8_max_abs_delta': int(np.max(np.abs(qref - qnew))),
    }
    reports.append(row)
    print(json.dumps(row))
  out = {'schema': 'mla-npu-attn-numeric-v1', 'NPU_used': False,
         'geometry': {'layers': 62, 'heads': HEADS, 'rows': ROWS, 'ctx': CTX, 'latent': LATENT,
                      'nope': NOPE, 'rope': ROPE, 'v_dim': VDIM},
         'scope': 'Synthetic activations at real shapes; FP16 operands + FP32 accumulation + exact-order '
                  'CPU softmax + FP16 probability operand vs the shipped absorbed FP32 CPU reference. '
                  'No model/quality claim; GGUF weights unavailable offline.',
         'reports': reports}
  args.json.write_text(json.dumps(out, indent=2) + '\n')
  print(f'wrote {args.json}')


if __name__ == '__main__':
  main()
