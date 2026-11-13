"""Opt-in vendor-style FP16 NPU MLA **prefill** attention (graph-level form).

MiniCPM3-4B prefill only.  Replaces the exact-FP32 CPU absorbed MLA attention and the two
absorbed CPU projections with the vendor's unabsorbed form:

  k_nope = c_kv @ W_kb,  v = c_kv @ W_vb          (grouped FP16, static weights)
  scores = q @ [k_nope|k_rope]^T                  (grouped FP16, dynamic K rhs)
  P      = exact-order causal softmax(scores)     (CPU, shipped fused-kernel arithmetic)
  ctx    = P @ v                                  (grouped FP16, dynamic V rhs)
  out    = token-major(ctx)                       (CPU)

All four matmuls use the already-wired `grouped_matmul(..., dynamic_weights=True, row_major=True)`
(version-8 dynamic row-major FP16 plans, k<=512/n<=512/m<=128). The K/V dynamic rhs and the
row-major lhs packs come from `tinygrad.runtime.support.rockchip_mla_npu_pack` (stream-diag task-25).

Decode and any non-matching geometry fall back to the shipped CPU absorbed path unchanged; with
`ONPU_ATTN_FP16_NPU=0` nothing is installed and the graph is byte-identical to today.
"""
from __future__ import annotations
import inspect
from tinygrad import Tensor, dtypes
from tinygrad.helpers import ContextVar
from tinygrad.llm.kernels.rockchip import grouped_matmul, pack_grouped_weight

ONPU_ATTN_FP16_NPU = ContextVar('ONPU_ATTN_FP16_NPU', 0)


def _static_start(start:int|object) -> int|None:
  """Resolve a bound/constant position UOp to its captured integer (prefill binds start_pos to 0)."""
  if type(start) is int: return start
  from tinygrad.uop.ops import Ops, UOp
  if isinstance(start,UOp):
    if start.op is Ops.CONST: return int(start.arg)
    if start.op is Ops.PARAM and getattr(start.arg,'val',None) is not None: return int(start.arg.val)
  return None

# Qualified geometry (the only MLA geometry MiniCPM3-4B ships).
_HEADS, _CONTEXT, _LATENT, _ROPE, _NOPE, _V_DIM = 40, 512, 256, 32, 64, 64
_WIDTH, _CAPACITY = _LATENT + _ROPE, 128


def _grouped_fp16(lhs:Tensor, rhs:Tensor, columns:int, *, static_rhs:bool=False) -> Tensor:
  """grouped_matmul with the version-8/9 dynamic row-major FP16 ABI, tolerant of a not-yet-patched tree."""
  kwargs:dict[str,object]={'cores':3,'dynamic_weights':not static_rhs,'row_major':True}
  if 'validate_finite' in inspect.signature(grouped_matmul).parameters:
    kwargs['validate_finite']=False  # ABI 10: skip the per-replay weight scan once stream-diag's patch lands
  try:
    return grouped_matmul(lhs, rhs, columns, **kwargs)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    if static_rhs:  # ABI 9 (static row-major) not present yet: the scan-paying dynamic path is equivalent
      return grouped_matmul(lhs, rhs, columns, cores=3, dynamic_weights=True, row_major=True)
    raise


class MLANPUAttention:
  """Per-block prefill attention hook. Holds the packed static expansion weights of one layer."""
  def __init__(self, key_b:Tensor, value_b:Tensor, pool=None):
    if (not isinstance(key_b,Tensor) or not isinstance(value_b,Tensor)
        or key_b.shape!=(_HEADS,_LATENT,_NOPE) or value_b.shape!=(_HEADS,_V_DIM,_LATENT)
        or key_b.device!='CPU' or value_b.device!='CPU'
        or key_b.dtype is not dtypes.half or value_b.dtype is not dtypes.half):
      raise ValueError('Unsupported MLA expansion weights')
    # grouped weight is (groups, columns, depth): k_nope = c_kv @ W_kb with W_kb[kv_lora_rank, nope]
    self.heads,self.capacity,self.context = _HEADS,_CAPACITY,_CONTEXT
    self.nope,self.rope,self.v_dim,self.latent = _NOPE,_ROPE,_V_DIM,_LATENT
    self.pool=pool
    self.kb = pack_grouped_weight(key_b.transpose(1,2).contiguous().realize())  # (heads, nope, latent)
    self.vb = pack_grouped_weight(value_b.contiguous().realize())               # (heads, v_dim, latent)

  def install_name(self) -> str: return 'npu_mla_prefill_attn'

  def _pool_address(self) -> int:
    """Parked prefill-pool dispatch for the softmax pack; 0 keeps the shipped serial path."""
    pool=self.pool
    # The parked endpoint is safe inside and outside the runtime scope (see CPUMLAProjections);
    # do NOT gate on pool.active here -- the hook runs inside a worker thread during capture.
    return int(pool.prefill_address) if pool is not None and self.heads>=4 and self.capacity>=16 else 0

  def __call__(self, q_nope:Tensor, q_rope:Tensor, c_kv:Tensor, k_rope:Tensor, start:int, rows:int) -> Tensor|None:
    """Return token-major (1, rows, heads*v_dim) FP32, or None to request the shipped CPU path."""
    # MVP guard set: single full chunk (the benchmark prompt is one 128-token prefill chunk).
    # The traced prefill graph carries a SYMBOLIC token dim, so row/shape checks must not
    # evaluate symbolic comparisons: validate through max_shape and let the pack kernels take
    # the fixed qualified extent (capacity). Otherwise the hook falls back to the CPU path.
    rows_in=rows
    if not isinstance(rows,int): rows=self.capacity
    if (rows!=self.capacity or _static_start(start)!=0
        or q_nope.device!='CPU' or q_rope.device!='CPU' or c_kv.device!='CPU' or k_rope.device!='CPU'
        or q_nope.dtype is not dtypes.float or q_rope.dtype is not dtypes.float
        or c_kv.dtype is not dtypes.float or k_rope.dtype is not dtypes.float
        or q_nope.ndim!=4 or q_rope.ndim!=4
        or q_nope.shape[:2]!=(1,self.heads) or q_rope.shape[:2]!=(1,self.heads)
        or not isinstance(q_nope.shape[-1],int) or q_nope.shape[-1]!=self.nope
        or not isinstance(q_rope.shape[-1],int) or q_rope.shape[-1]!=self.rope
        or int(q_nope.max_shape[2])!=self.capacity or int(q_rope.max_shape[2])!=self.capacity
        or int(c_kv.max_shape[1])!=self.capacity or int(k_rope.max_shape[2])!=self.capacity
        or int(c_kv.max_shape[2])!=self.latent or int(k_rope.max_shape[3])!=self.rope
        or tuple(c_kv.shape)[0]!=1 or tuple(k_rope.shape)[:2]!=(1,1)):
      return None
    from tinygrad.llm.kernels.rockchip_padding import capacity_input
    from tinygrad.runtime.support import rockchip_mla_npu_pack as pack

    dense_q = capacity_input(q_nope).contiguous()
    dense_r = capacity_input(q_rope).contiguous()
    dense_k = capacity_input(c_kv).contiguous()
    dense_kr = capacity_input(k_rope).contiguous()
    def _opt_call(fn, *args):
      """Accept both (..., capacity) and (...,) pack signatures while the pack module lands."""
      try: return fn(*args, self.capacity)
      except TypeError: return fn(*args)
    qa = _opt_call(pack.mla_npu_pack_query, dense_q, dense_r, rows)                  # (heads, cap, 96) fp16
    exp_lhs = _opt_call(pack.mla_npu_pack_expand_lhs, dense_k, rows)                 # (heads, cap, 256) fp16
    k_nope = _grouped_fp16(exp_lhs, self.kb, self.nope, static_rhs=True)             # (heads, cap, 64) f32
    v_exp = _grouped_fp16(exp_lhs, self.vb, self.v_dim, static_rhs=True)             # (heads, cap, 64) f32
    ka = pack.mla_npu_pack_k(k_nope, dense_kr, self.capacity).to('ROCKCHIP')  # keys==rows==capacity (start=0)
    va = pack.mla_npu_pack_v(v_exp, self.capacity).to('ROCKCHIP')                    # (heads, 4, 4, 16, 32)
    scores = _grouped_fp16(qa, ka, self.capacity)                                    # (heads, cap, 128) f32
    try:
      prob = pack.mla_npu_pack_prob(scores, 0, rows, self._pool_address())       # (heads, cap, 128) fp16
    except TypeError:  # pack module without the pool-dispatch patch: shipped serial path
      prob = pack.mla_npu_pack_prob(scores, 0, rows)
    ctx = _grouped_fp16(prob, va, self.v_dim)                                        # (heads, cap, 64) f32
    out = pack.mla_npu_unpack_context(ctx, rows)                                     # (capacity, heads*64) f32
    result=out.reshape(1, self.capacity, self.heads*self.v_dim)
    # Keep the traced token dim symbolic: the surrounding graph carries `toks` as a PARAM,
    # so slicing with the original rows (an int or the symbolic T) preserves shape compatibility.
    return result[:,:rows_in,:] if not isinstance(rows_in,int) or rows_in!=self.capacity else result


def configure_prefill(model, pool=None) -> int:
  """Install the opt-in NPU prefill attention hook. Raises when the flag is set but nothing is eligible."""
  if not ONPU_ATTN_FP16_NPU: return 0
  from tinygrad.llm.model import MLATransformerBlock
  blocks=[b for b in getattr(model,'blk',[]) if isinstance(b,MLATransformerBlock)]
  installed=0
  for block in blocks:
    c=getattr(block,'config',None)
    if (c is None or (getattr(c,'n_heads',0),getattr(c,'kv_lora_rank',0),getattr(c,'rope_dim',0),getattr(c,'v_head_dim',0))
        !=(_HEADS,_LATENT,_ROPE,_V_DIM) or getattr(c,'max_context',0)>_CONTEXT):continue
    kb=block.attn_k_b.get('weight') if isinstance(getattr(block,'attn_k_b',None),dict) else None
    vb=block.attn_v_b.get('weight') if isinstance(getattr(block,'attn_v_b',None),dict) else None
    try:block.npu_mla_prefill_attn=MLANPUAttention(kb,vb,pool=pool)
    except (ValueError,TypeError):continue
    installed+=1
  if not installed:
    raise RuntimeError('ONPU_ATTN_FP16_NPU=1 but no eligible MLA transformer block was found '
                       '(need MiniCPM3-4B MLA geometry (40 heads, kv_lora 256, rope 32, v 64))')
  return installed
