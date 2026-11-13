"""Bit-exactness of the fused one-token QKV dequantize + partial-rotary + cache kernel.

The one-token decode graph currently launches, per layer:
  * one CPU dequantize kernel over the projected INT8 QKV row,
  * one elementwise query-rotary kernel (E_32_32_2_2, serialized on 32 lanes),
  * one elementwise key-rotary + value-store kernel (E_2_32_2_2_2).
`prepare_qkv_int8(..., rope_dim=...)` replaces all three with a single host
program. These tests compare that fused kernel bit-for-bit (`np.array_equal`,
`max|diff| == 0`) against the generic graph it replaces, for ChatGLM-style
partial rotary (rope_dim = head_dim/2), including token 0 and the last context
slot.
"""
import os, re, shutil, subprocess, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, function
from tinygrad.llm.kernels.rockchip_qkv import prepare_qkv, prepare_qkv_int8
from tinygrad.llm.model import apply_rope, precompute_freqs_cis
from tinygrad.helpers import ceildiv, round_up


def clang_assembly(source:str) -> str:
  """Compile emitted kernel C to assembly with the CPU backend's real clang flags."""
  clang = shutil.which(os.getenv('CC','clang'))
  if clang is None: raise unittest.SkipTest('clang not available')
  from tinygrad.llm.kernels.rockchip_cpu import _renderer
  renderer = _renderer()
  arch = str(getattr(renderer.target,'arch','')).split(',')[0]
  if not arch: raise unittest.SkipTest('CPU renderer has no arch')
  flags = [clang,'-S','-x','c','-O2','-fPIC','-ffreestanding','-fno-math-errno','-nostdlib','-fno-ident',
           f'--target={arch}-none-unknown-elf', *renderer.compiler.args, '-o','-','-']
  return subprocess.check_output(flags,input=source.encode(),stderr=subprocess.PIPE).decode()


def has_fused_multiply_add(text:str) -> bool:
  """True when the compiler emitted a fused multiply-add for a product/sum chain."""
  return bool(re.search(r'\b(fmla|fmls|fmadd|fmsub|vfmadd|vfnmadd|fnmadd)', text))


def program_sources(tensor:Tensor) -> list[str]:
  """Distinct emitted C sources of the host programs a tensor's graph lowers to."""
  from tinygrad.engine.realize import compile_linear
  from tinygrad.uop.ops import Ops
  out:list[str] = []
  for call in compile_linear(tensor.schedule_linear()).toposort():
    if call.op.name!='CALL' or call.src[0].op.name!='PROGRAM': continue
    src = next((u.arg for u in call.src[0].src if u.op is Ops.SOURCE), None)
    if src and src not in out: out.append(src)
  return out


def reference_decode_qkv(projected:Tensor, cache:Tensor, freqs:Tensor, start:int, heads:int, kv_heads:int,
                         dim:int, rope_dim:int) -> tuple[Tensor,Tensor]:
  """The generic decode graph from `TransformerBlock._attention` (verbatim ops)."""
  count = projected.shape[1]
  qv = projected[:, :, :heads*dim].reshape(1, count, heads, dim).transpose(1, 2)
  kvw = projected[:, :, heads*dim:(heads+kv_heads)*dim].reshape(1, count, kv_heads, dim).transpose(1, 2)
  vv = projected[:, :, (heads+kv_heads)*dim:].reshape(1, count, kv_heads, dim).transpose(1, 2)
  q = apply_rope(qv[..., :rope_dim], freqs[start:start+count]).cat(qv[..., rope_dim:], dim=-1)
  k = apply_rope(kvw[..., :rope_dim], freqs[start:start+count]).cat(kvw[..., rope_dim:], dim=-1)
  store = cache[:, :, :, start:start+count, :].uop.store(Tensor.stack(k, vv).cast(cache.dtype).uop)
  return q, Tensor(cache.uop.after(store))


def raw_3chunk(values:np.ndarray, columns:int) -> np.ndarray:
  """Pack an int32 (rows, columns) matrix the way the 3-core NPU INT8 matmul emits it."""
  rows = values.shape[0]
  padded = round_up(columns, 32)
  width = round_up(ceildiv(padded, 3), 32)
  parts = [(first, min(width, padded-first)) for first in range(0, padded, width)]
  if len(parts) != 3: parts = [(0, padded)]
  out = np.zeros((rows*sum(size for _, size in parts),), dtype=np.int32)
  offset = 0
  for first, size in parts:
    chunk = values[:, first:first+size]
    out[offset:offset+rows*size] = chunk.reshape(-1)
    offset += rows*size
  return out


class TestDecodeFusedQKVRope(unittest.TestCase):
  heads, kv_heads, dim, rope_dim, context = 32, 2, 128, 64, 512
  columns = (heads + 2*kv_heads)*dim

  def _geometry(self):
    return self.heads, self.kv_heads, self.dim, self.rope_dim, self.context, self.columns

  def test_dense_partial_rotary_matches_generic_graph(self):
    rng = np.random.default_rng(20261109)
    heads, kv, dim, rope, context, columns = self._geometry()
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    for count, start in ((1, 0), (1, context-1), (4, 7), (17, 5), (16, context-16)):
      data = rng.normal(0, 0.8, (1, count, columns)).astype(np.float32)
      initial = rng.normal(0, 0.2, (2, 1, kv, context, dim)).astype(np.float16)
      cache_ref = Tensor(initial.copy(), device='CPU').realize()
      cache_got = Tensor(initial.copy(), device='CPU').realize()
      projected = Tensor(data, device='CPU').realize()
      expected = reference_decode_qkv(projected, cache_ref, freqs, start, heads, kv, dim, rope)
      Tensor.realize(*expected)
      got_q, got_kv = prepare_qkv(projected, cache_got, None, None, freqs, start, heads, 1e-6, rope_dim=rope)
      Tensor.realize(got_q, got_kv)
      # FP32 query rows: must be bit-identical, not merely close.
      np.testing.assert_array_equal(got_q.numpy(), expected[0].numpy())
      # Key/value FP16 cache slots: compare raw half bits.
      np.testing.assert_array_equal(got_kv.numpy().view(np.uint16), expected[1].numpy().view(np.uint16))

  def test_int8_fused_matches_generic_graph(self):
    """End to end: mock NPU raw QKV -> fused kernel vs generic dequantize + rope + store."""
    from tinygrad.llm.kernels.rockchip_int8 import dequantize_tiles
    rng = np.random.default_rng(20261110)
    heads, kv, dim, rope, context, columns = self._geometry()
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    sw = rng.uniform(.0005, .002, columns).astype(np.float32)
    for count, start in ((1, 0), (1, context-1), (1, 300)):
      values = rng.integers(-120000, 120000, (count, columns), dtype=np.int32)
      sa = rng.uniform(.001, .002, count).astype(np.float32)
      raw = Tensor(raw_3chunk(values, columns), device='CPU').realize()
      scale = Tensor(sa, device='CPU').realize()
      sw_t = Tensor(sw, device='CPU').realize()
      # Generic graph: the real CPU dequantize kernel, then the model's movements.
      dense = dequantize_tiles([raw], [scale], sw_t, [count], columns)
      dense = dense.reshape(1, count, columns)
      initial = rng.normal(0, 0.2, (2, 1, kv, context, dim)).astype(np.float16)
      cache_ref = Tensor(initial.copy(), device='CPU').realize()
      cache_got = Tensor(initial.copy(), device='CPU').realize()
      expected = reference_decode_qkv(dense, cache_ref, freqs, start, heads, kv, dim, rope)
      Tensor.realize(*expected)
      got_q, got_kv = prepare_qkv_int8([raw], [scale], sw_t, [count], cache_got, None, None, freqs, start, heads, 1e-6,
                                       count, rope_dim=rope)
      Tensor.realize(got_q, got_kv)
      np.testing.assert_array_equal(got_q.numpy(), expected[0].numpy())
      np.testing.assert_array_equal(got_kv.numpy().view(np.uint16), expected[1].numpy().view(np.uint16))

  def test_int8_fused_matches_dense_prepare(self):
    """Both prepared paths must agree without normalization (the ChatGLM case)."""
    rng = np.random.default_rng(20261111)
    heads, kv, dim, rope, context, columns = self._geometry()
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    for count, start in ((1, 0), (7, 3)):
      values = rng.integers(-120000, 120000, (count, columns), dtype=np.int32)
      sa = rng.uniform(.001, .002, count).astype(np.float32)
      sw = rng.uniform(.0005, .002, columns).astype(np.float32)
      raw = Tensor(raw_3chunk(values, columns), device='CPU').realize()
      dense = Tensor((values.astype(np.float32)*sa[:, None]*sw[None, :])[None], device='CPU').realize()
      initial = rng.normal(0, .2, (2, 1, kv, context, dim)).astype(np.float16)
      cache_a, cache_b = (Tensor(initial.copy(), device='CPU').realize() for _ in range(2))
      expected = prepare_qkv(dense, cache_a, None, None, freqs, start, heads, 1e-6, rope_dim=rope)
      actual = prepare_qkv_int8([raw], [Tensor(sa, device='CPU')], Tensor(sw, device='CPU'), [count], cache_b,
                                None, None, freqs, start, heads, 1e-6, count, rope_dim=rope)
      np.testing.assert_array_equal(actual[0].numpy(), expected[0].numpy())
      np.testing.assert_array_equal(actual[1].numpy().view(np.uint16), expected[1].numpy().view(np.uint16))

  def test_symbolic_start_and_count_rebinding(self):
    """JIT replay passes start/rows as device pointers, not immediates."""
    rng = np.random.default_rng(20261112)
    heads, kv, dim, rope, context, columns = self._geometry()
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    values = rng.integers(-120000, 120000, (1, columns), dtype=np.int32)
    sa = np.array([0.0015], dtype=np.float32)
    sw = rng.uniform(.0005, .002, columns).astype(np.float32)
    raw = Tensor(raw_3chunk(values, columns), device='CPU').realize()
    initial = rng.normal(0, .2, (2, 1, kv, context, dim)).astype(np.float16)

    @function(precompile=True)
    def prepare(raw:UOp, scale:UOp, sw:UOp, cache:UOp, freqs:UOp, start:UOp):
      return prepare_qkv_int8([raw], [scale], sw, [1], cache, None, None, freqs, start, heads, 1e-6, 1, rope_dim=rope)

    @TinyJit
    def run(raw:Tensor, scale:Tensor, sw:Tensor, cache:Tensor, freqs:Tensor, start:UOp):
      # start is symbolic inside the traced graph, so binding it exercises the pointer ABI.
      result = prepare(raw, scale, sw, cache, freqs, start)
      Tensor.realize(*result)
      return result

    for start in (0, context-1, 271):
      cache_got = Tensor(initial.copy(), device='CPU').realize()
      cache_ref = Tensor(initial.copy(), device='CPU').realize()
      got_q, got_kv = run(raw, Tensor(sa, device='CPU'), Tensor(sw, device='CPU'), cache_got, freqs,
                          UOp.variable('start', 0, context-1).bind(start))
      dense = Tensor((values.astype(np.float32)*sa[:, None]*sw[None, :])[None], device='CPU')
      expected = reference_decode_qkv(dense, cache_ref, freqs, start, heads, kv, dim, rope)
      Tensor.realize(*expected)
      np.testing.assert_array_equal(got_q.numpy()[:, :, :1], expected[0].numpy())
      np.testing.assert_array_equal(got_kv.numpy().view(np.uint16), expected[1].numpy().view(np.uint16))

  def test_full_rotary_default_is_unchanged(self):
    """rope_dim=0 (default) must reproduce the original full-head rotary bytes."""
    rng = np.random.default_rng(20261113)
    heads, kv, dim, context = 16, 8, 128, 64
    columns = (heads + 2*kv)*dim
    freqs = precompute_freqs_cis(dim, context, device='CPU').realize()
    data = rng.normal(0, .8, (1, 5, columns)).astype(np.float32)
    initial = rng.normal(0, .2, (2, 1, kv, context, dim)).astype(np.float16)
    cache_a, cache_b = (Tensor(initial.copy(), device='CPU').realize() for _ in range(2))
    projected = Tensor(data, device='CPU').realize()
    default = prepare_qkv(projected, cache_a, None, None, freqs, 9, heads, 1e-6)
    explicit = prepare_qkv(projected, cache_b, None, None, freqs, 9, heads, 1e-6, rope_dim=dim)
    np.testing.assert_array_equal(default[0].numpy(), explicit[0].numpy())
    np.testing.assert_array_equal(default[1].numpy().view(np.uint16), explicit[1].numpy().view(np.uint16))

  def test_partial_rotary_rejects_bad_geometry(self):
    heads, kv, dim, rope, context = self.heads, self.kv_heads, self.dim, self.rope_dim, self.context
    columns = self.columns
    projected = Tensor.zeros(1, 1, columns, device='CPU').realize()
    cache = Tensor.zeros(2, 1, kv, context, dim, device='CPU', dtype=dtypes.half).realize()
    full_freqs = Tensor.zeros(context, dim, device='CPU').realize()
    short_freqs = Tensor.zeros(context, rope, device='CPU').realize()
    with self.assertRaises(ValueError): prepare_qkv(projected, cache, None, None, short_freqs, 0, heads, 1e-6, rope_dim=rope+1)
    with self.assertRaises(ValueError): prepare_qkv(projected, cache, None, None, short_freqs, 0, heads, 1e-6, rope_dim=dim+1)
    with self.assertRaises(ValueError): prepare_qkv(projected, cache, None, None, full_freqs, 0, heads, 1e-6, rope_dim=rope)


  # ---------------------------------------------------------------- audit item 2
  def test_partial_rotary_t1_q_and_cache_match_model_apply_rope(self):
    """rope_dim=64 / head_dim=128 / T=1: Q must equal model.apply_rope exactly and the
    KV cache FP16 bytes must equal the generic stack(k,v).cast(half) store."""
    rng = np.random.default_rng(20261114)
    heads, kv, dim, rope, context = 32, 2, 128, 64, 512
    columns = (heads + 2*kv)*dim
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    for start in (0, 1, 200, context-1):
      # Random Q/K/V for one token (T=1), laid out as the fused QKV projection output.
      q_raw = rng.normal(0, 1.3, (1, 1, heads, dim)).astype(np.float32)
      k_raw = rng.normal(0, 1.1, (1, 1, kv, dim)).astype(np.float32)
      v_raw = rng.normal(0, 1.0, (1, 1, kv, dim)).astype(np.float32)
      projected = np.concatenate([q_raw.reshape(1, 1, heads*dim), k_raw.reshape(1, 1, kv*dim),
                                  v_raw.reshape(1, 1, kv*dim)], axis=-1).astype(np.float32)
      # Reference = model.apply_rope + cat + stack + fp16 store, exactly as _attention does it.
      q4 = Tensor(q_raw, device='CPU').transpose(1, 2)
      k4 = Tensor(k_raw, device='CPU').transpose(1, 2)
      v4 = Tensor(v_raw, device='CPU').transpose(1, 2)
      q_ref = apply_rope(q4[..., :rope], freqs[start:start+1]).cat(q4[..., rope:], dim=-1)
      k_ref = apply_rope(k4[..., :rope], freqs[start:start+1]).cat(k4[..., rope:], dim=-1)
      initial = rng.normal(0, 0.2, (2, 1, kv, context, dim)).astype(np.float16)
      cache_ref = Tensor(initial.copy(), device='CPU').realize()
      cache_got = Tensor(initial.copy(), device='CPU').realize()
      store = cache_ref[:, :, :, start:start+1, :].uop.store(Tensor.stack(k_ref, v4).cast(cache_ref.dtype).uop)
      Tensor.realize(q_ref, Tensor(cache_ref.uop.after(store)))
      got_q, got_kv = prepare_qkv(Tensor(projected, device='CPU'), cache_got, None, None, freqs, start, heads, 1e-6,
                                  rope_dim=rope)
      Tensor.realize(got_q, got_kv)
      np.testing.assert_array_equal(got_q.numpy(), q_ref.numpy())                       # FP32 query bytes
      np.testing.assert_array_equal(got_kv.numpy().view(np.uint16),
                                    cache_ref.numpy().view(np.uint16))                  # FP16 cache bytes

  def test_full_rotary_matches_generic_graph(self):
    """rope_dim=0 (full-head rotary) must also match the generic path bit-for-bit."""
    rng = np.random.default_rng(20261115)
    for heads, kv, dim, context, count, start in ((16, 8, 128, 64, 1, 9), (16, 8, 128, 64, 5, 3), (4, 2, 64, 32, 1, 0)):
      columns = (heads + 2*kv)*dim
      freqs = precompute_freqs_cis(dim, context, device='CPU').realize()   # full rotary: rope tables are head-wide
      data = rng.normal(0, .8, (1, count, columns)).astype(np.float32)
      initial = rng.normal(0, .2, (2, 1, kv, context, dim)).astype(np.float16)
      cache_ref = Tensor(initial.copy(), device='CPU').realize()
      cache_got = Tensor(initial.copy(), device='CPU').realize()
      projected = Tensor(data, device='CPU').realize()
      expected = reference_decode_qkv(projected, cache_ref, freqs, start, heads, kv, dim, dim)
      Tensor.realize(*expected)
      got_q, got_kv = prepare_qkv(projected, cache_got, None, None, freqs, start, heads, 1e-6)
      Tensor.realize(got_q, got_kv)
      np.testing.assert_array_equal(got_q.numpy(), expected[0].numpy())
      np.testing.assert_array_equal(got_kv.numpy().view(np.uint16), expected[1].numpy().view(np.uint16))

  # ---------------------------------------------------------------- audit item 1
  def test_rope_fma_rounding_matches_generic_kernel(self):
    """The fused rope must use the SAME fused rounding as the generic tinygrad kernel.

    Compile both kernels with the CPU backend's real clang flags and require a fused
    multiply-add in both. tinygrad lowers `(x1*cos)-(x2*sin)` with C's default
    -ffp-contract=on to FMLA, so an unfused kernel is 1 ulp off: the fused kernel must
    not rely on the contraction heuristic, it emits the FMA explicitly and keeps the
    two standalone products under `#pragma clang fp contract(off)`.
    """
    heads, kv, dim, rope, context = 32, 2, 128, 64, 512
    columns = (heads + 2*kv)*dim
    freqs = precompute_freqs_cis(rope, context, device='CPU').realize()
    rng = np.random.default_rng(20261116)
    raw = Tensor(raw_3chunk(rng.integers(-120000, 120000, (1, columns), dtype=np.int32), columns), device='CPU').realize()
    sa = Tensor(np.array([0.0015], dtype=np.float32), device='CPU').realize()
    sw = Tensor(rng.uniform(.0005, .002, columns).astype(np.float32), device='CPU').realize()
    cache = Tensor.zeros(2, 1, kv, context, dim, device='CPU', dtype=dtypes.half).realize()
    fused_q, _ = prepare_qkv_int8([raw], [sa], sw, [1], cache, None, None, freqs, 0, heads, 1e-6, 1, rope_dim=rope)
    fused_src = [s for s in program_sources(fused_q) if 'rockchip_prepare_qkv' in s]
    self.assertTrue(fused_src, 'no fused QKV program found')
    # The generic decode rope graph: the exact q/k rope+cat from TransformerBlock._attention.
    columns = (heads + 2*kv)*dim
    dense = Tensor.rand(1, 1, columns, device='CPU')
    qv = dense[:, :, :heads*dim].reshape(1, 1, heads, dim).transpose(1, 2)
    kvw = dense[:, :, heads*dim:(heads+kv)*dim].reshape(1, 1, kv, dim).transpose(1, 2)
    vv = dense[:, :, (heads+kv)*dim:].reshape(1, 1, kv, dim).transpose(1, 2)
    q_ref = apply_rope(qv[..., :rope], freqs[0:1]).cat(qv[..., rope:], dim=-1)
    k_ref = apply_rope(kvw[..., :rope], freqs[0:1]).cat(kvw[..., rope:], dim=-1)
    store = cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k_ref, vv).cast(cache.dtype).uop)
    generic_sources = program_sources(q_ref) + program_sources(Tensor(cache.uop.after(store)))
    generic_src = [s for s in generic_sources if re.search(r'E_(?:2_)?32_32_2(?:_2)?\(', s)]
    self.assertTrue(generic_src, f'no generic rope program found among {[s[:40] for s in generic_sources]}')
    fused_src, generic_src = fused_src[0], generic_src[0]
    # The emitted C must contain the explicit fused rounding, not a bare product/sum chain.
    self.assertIn('fmla', fused_src)
    self.assertIn('contract(off)', fused_src)
    fused_asm = clang_assembly(fused_src)
    generic_asm = clang_assembly(generic_src)
    self.assertTrue(has_fused_multiply_add(fused_asm), 'fused kernel assembly has no fused multiply-add')
    self.assertTrue(has_fused_multiply_add(generic_asm),
                    'generic tinygrad rope kernel did not contract: re-derive the rounding before trusting bit-exactness')


if __name__ == '__main__':
  unittest.main()
