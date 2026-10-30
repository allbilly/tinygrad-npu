import numpy as np
from tinygrad import Context, Tensor, TinyJit, UOp
from tinygrad.uop.ops import Ops
from tinygrad.llm.kernels.rockchip_cpu import prefill_attention
from tinygrad.llm.kernels.rockchip_padding import capacity_input, capacity_view_counts, mark_zero_tail

HEADS,KV_HEADS,DIM,CONTEXT,CAPACITY=32,4,64,512,128


def _bits(x):return np.asarray(x).view(np.uint32)


def _run():
  def forward(query,cache,total,rows):
    active=mark_zero_tail(query[:,:,:rows,:],query,'qkv')
    output=prefill_attention(active,cache,total,fast_softmax=True)
    return capacity_input(output).realize()
  return TinyJit(forward)


def test_token_major_prefill_layout_matches_head_major_and_zeroes_dynamic_tail():
  rng=np.random.default_rng(20261102)
  baseline=_run()
  token_major=_run()
  view_before=capacity_view_counts().get('cpu_attention',0)
  view_after_previous=view_before
  with Context(ONPU_ZERO_PAD_VIEWS=1,ONPU_CPU_PREFILL_LAYOUT=0):
    for index,(rows,total) in enumerate(((128,128),(17,190),(65,512),(1,128))):
      qdata=rng.normal(0,.3,(1,HEADS,CAPACITY,DIM)).astype(np.float32)
      qdata[:,:,rows:]=0
      cache_data=rng.normal(0,.2,(2,1,KV_HEADS,CONTEXT,DIM)).astype(np.float16)
      cache_data[:,:,:,total:]=np.nan
      query=Tensor(qdata,device='CPU').realize()
      cache=Tensor(cache_data,device='CPU').realize()
      row_var=UOp.variable('prefill_rows',1,CAPACITY).bind(rows)
      total_var=UOp.variable('prefill_total',1,CONTEXT).bind(total)
      with Context(ONPU_ZERO_PAD_VIEWS=1,ONPU_CPU_PREFILL_LAYOUT=0):
        reference=baseline(query,cache,total_var,row_var)
        expected=reference.numpy().copy()
      assert expected.shape==(1,HEADS,CAPACITY,DIM)
      assert np.isfinite(expected[:,:,:rows]).all()
      assert not expected[:,:,rows:].any()
      with Context(ONPU_ZERO_PAD_VIEWS=1,ONPU_CPU_PREFILL_LAYOUT=1):
        full=token_major(query,cache,total_var,row_var)
        physical=full.transpose(1,2).reshape(1,CAPACITY,HEADS*DIM)
        got=full.numpy()
        flat=physical.numpy()
      np.testing.assert_array_equal(_bits(got[:,:,:rows]),_bits(expected[:,:,:rows]))
      assert not got[:,:,rows:].any()
      np.testing.assert_array_equal(_bits(flat[:,:rows]),_bits(expected[:,:,:rows].transpose(0,2,1,3).reshape(1,rows,HEADS*DIM)))
      assert physical.shape==(1,CAPACITY,HEADS*DIM)
      assert physical.uop.op is Ops.RESHAPE
      assert physical.uop.src[0].op is Ops.PERMUTE
      views=capacity_view_counts().get('cpu_attention',0)
      if index==0:assert views>view_before
      else:assert views>=view_after_previous
      view_after_previous=views
