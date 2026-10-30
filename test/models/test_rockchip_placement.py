from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
import pytest
from tinygrad import Tensor, nn
from tinygrad.uop.ops import UOp
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig
from tinygrad.llm.kernels.rockchip import RockchipLinear, configure


class _FakePool:
  def __init__(self):
    self.active=False
    self.prefill_entries=0
    self.entries=0
  def __enter__(self):
    self.active=True
    self.entries+=1
    return self
  def __exit__(self,*args):
    self.active=False
  @contextmanager
  def prefill(self):
    self.active=True
    self.prefill_entries+=1
    try: yield self
    finally: self.active=False


def _cpu_gdn_model(*,context=512,n_kv_heads=1,num_v_heads=4):
  ssm=SSMConfig(conv_kernel=2,state_size=8,group_count=1,time_step_rank=num_v_heads,inner_size=8*num_v_heads)
  return Transformer(TransformerConfig(1,16,32,2,n_kv_heads,1e-5,32,8,10000.,8,8,
                                      max_context=context,ssm=ssm,ssm_layers=(True,)))


def _check_gated_attention(monkeypatch,precision,gates,fast_softmax,sliding_window=0,rows=(17,1,7)):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  output_gate,head_gate = gates
  Tensor.manual_seed(41)
  model = Transformer(TransformerConfig(1,32,64,4,2,1e-6,32,8,10000.,8,8,max_context=64,qk_norm=8,
                                        attn_output_gate=output_gate,attn_head_gate=head_gate,sliding_window=sliding_window))
  block = model.blk[0]
  inputs = [Tensor.randn(1,count,32,device='CPU').realize() for count in rows]
  block._init_state(inputs[0])
  expected = []
  start = 0
  for x in inputs:
    expected.append(block._attention(x,start).numpy().copy())
    start += x.shape[1]
  block.cache_kv.assign(Tensor.zeros_like(block.cache_kv)).realize()
  # Isolate attention math and placement from NPU quantization/driver access.
  for cls in (RockchipLinear,RockchipInt8Linear):
    monkeypatch.setattr(cls,'prepare',lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU')]))
    monkeypatch.setattr(cls,'npu',lambda self,x:self.source(x))
  result = configure(model,prefill='on',decode='on',cpu_attention=True,fuse_projections=True,
                     fused_qkv=True,precision=precision,cpu_fast_softmax=fast_softmax)
  assert result['cpu_attention_blocks']==1
  assert result['prepared_qkv_blocks']==0 and not hasattr(block,'prepared_qkv')
  start = 0
  for x,reference in zip(inputs,expected):
    np.testing.assert_allclose(block._attention(x,start).numpy(),reference,atol=2e-5,rtol=2e-4)
    start += x.shape[1]


@pytest.mark.parametrize('precision',['fp16','int8'])
@pytest.mark.parametrize('gates',[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('fast_softmax',[False,True])
def test_gated_attention_reuses_cpu_kernel_and_preserves_cache_history(monkeypatch,precision,gates,fast_softmax):
  _check_gated_attention(monkeypatch,precision,gates,fast_softmax)


@pytest.mark.parametrize('window',[64,128])
def test_window_covering_full_context_reuses_causal_kernel(monkeypatch,window):
  _check_gated_attention(monkeypatch,'int8',(False,True),True,sliding_window=window,rows=(17,1,7,39))


def test_packed_attention_preserves_window_covering_allocated_context(monkeypatch):
  Tensor.manual_seed(46)
  model = Transformer(TransformerConfig(1,576,1152,9,3,1e-6,32,64,10000.,64,64,
                                       max_context=4096,sliding_window=4096))
  block = model.blk[0]
  x = Tensor.randn(1,17,576,device='CPU').realize()
  block._init_state(x)
  expected = block._attention(x,0).numpy().copy()
  block.cache_kv.assign(Tensor.zeros_like(block.cache_kv)).realize()
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU')]))
  monkeypatch.setattr(RockchipLinear,'npu',lambda self,x:self.source(x))
  placement = configure(model,prefill='on',decode='on',cpu_attention=True,cpu_packed_attention=True,cpu_fast_softmax=True)
  assert placement['packed_attention_blocks']==1
  np.testing.assert_allclose(block._attention(x,0).numpy(),expected,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('window',[1,8,17])
@pytest.mark.parametrize('fast_softmax',[False,True])
def test_sliding_window_reuses_cpu_kernel_across_cached_calls(monkeypatch,window,fast_softmax):
  _check_gated_attention(monkeypatch,'int8',(True,True),fast_softmax,sliding_window=window)


@pytest.mark.parametrize('unsupported',[{'attn_sinks':True},{'attn_logit_softcap':129.0},{'max_context':8193}])
def test_gated_attention_keeps_unsupported_features_on_generic_path(unsupported):
  options = {'max_context':64,'attn_output_gate':True,**unsupported}
  model = Transformer(TransformerConfig(1,32,64,4,2,1e-6,32,8,10000.,8,8,**options))
  result = configure(model,prefill='off',decode='off',cpu_attention=True)
  assert result['cpu_attention_blocks']==0 and result['prepared_qkv_blocks']==0


@pytest.mark.parametrize('precision',['fp16','int8'])
@pytest.mark.parametrize('context',[4096,8192])
def test_prepared_qkv_keeps_larger_context_on_regular_cpu_attention(monkeypatch,precision,context):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  def prepare(self): self.packed = [Tensor.empty(1,device='CPU')]
  for cls in (RockchipLinear,RockchipInt8Linear):
    monkeypatch.setattr(cls,'prepare',prepare)
    monkeypatch.setattr(cls,'npu',lambda self,x:self.source(x))
  model = Transformer(TransformerConfig(1,32,64,4,2,1e-6,32,8,10000.,8,8,max_context=context,qk_norm=8))
  result = configure(model,prefill='on',cpu_attention=True,fuse_projections=True,fused_qkv=True,precision=precision)
  block = model.blk[0]
  assert result['cpu_attention_blocks']==1
  assert result['prepared_qkv_blocks']==int(context<=4096)
  x = Tensor.rand(1,16,32,device='CPU').realize()
  if context<=4096:
    assert block.prepared_qkv.enabled(x)
  else:
    assert not hasattr(block,'prepared_qkv')
    # Exercise the actual regular CPU prefill after configuration, including its
    # full-size KV allocation. No patched path initializes Rockchip hardware.
    block._init_state(x)
    output = block._attention(x,0).numpy()
    assert output.shape==(1,16,32) and np.isfinite(output).all()


@pytest.mark.parametrize('precision,bias',[('fp16',False),('fp16',True),('int8',True)])
def test_unnormalized_prepared_qkv_preserves_model_attention_and_cache(monkeypatch,precision,bias):
  from tinygrad.llm.kernels.rockchip import RockchipPreparedQKV
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8PreparedQKV
  Tensor.manual_seed(61)
  model = Transformer(TransformerConfig(1,32,64,4,2,1e-6,32,8,10000.,8,8,max_context=64,qkv_bias=bias))
  block = model.blk[0]
  values = [Tensor.randn(1,rows,32,device='CPU').realize() for rows in (17,1,7)]
  block._init_state(values[0])
  expected=[]
  start=0
  for value in values:
    expected.append((block._attention(value,start).numpy().copy(),block.cache_kv.numpy().copy()))
    start+=value.shape[1]
  block.cache_kv.assign(Tensor.zeros_like(block.cache_kv)).realize()
  for cls in (RockchipLinear,RockchipInt8Linear):
    monkeypatch.setattr(cls,'prepare',lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU')]))
    monkeypatch.setattr(cls,'npu',lambda self,x:self.source(x))
  # These placement fixtures replace device projections with their float CPU
  # source. The raw INT32 path has separate biased-QKV numerical tests.
  monkeypatch.setattr(RockchipInt8PreparedQKV,'__call__',RockchipPreparedQKV.__call__)
  result=configure(model,prefill='on',decode='on',cpu_attention=True,fuse_projections=True,fused_qkv=True,
                   fused_qkv_without_norm=True,precision=precision)
  assert result['prepared_qkv_blocks']==1 and block.prepared_qkv.qnorm is block.prepared_qkv.knorm is None
  start=0
  for value,(output,cache) in zip(values,expected):
    np.testing.assert_allclose(block._attention(value,start).numpy(),output,atol=2e-5,rtol=2e-4)
    np.testing.assert_allclose(block.cache_kv.numpy(),cache,atol=2e-3,rtol=2e-3)
    start+=value.shape[1]


@pytest.mark.parametrize('enabled,context,gated,prepared',[(False,64,False,0),(True,64,False,1),
                                                       (True,8192,False,0),(True,64,True,0)])
def test_unnormalized_prepared_qkv_remains_explicit_and_bounded(monkeypatch,enabled,context,gated,prepared):
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear, RockchipInt8PreparedQKV
  for cls in (RockchipLinear,RockchipInt8Linear):
    monkeypatch.setattr(cls,'prepare',lambda self:setattr(self,'packed',[Tensor.empty(1,device='CPU')]))
  model=Transformer(TransformerConfig(1,32,64,4,2,1e-6,32,8,10000.,8,8,max_context=context,attn_output_gate=gated))
  result=configure(model,prefill='on',cpu_attention=True,fuse_projections=True,fused_qkv=True,
                   fused_qkv_without_norm=enabled,precision='int8')
  assert result['prepared_qkv_blocks']==prepared
  if prepared:assert isinstance(model.blk[0].prepared_qkv,RockchipInt8PreparedQKV)


def test_unnormalized_qkv_requires_fused_qkv_selection():
  with pytest.raises(ValueError,match='requires fused_qkv'):configure(SimpleNamespace(),fused_qkv_without_norm=True)
  with pytest.raises(ValueError,match='must be a bool'):configure(SimpleNamespace(),fused_qkv_without_norm=1)


@pytest.mark.parametrize('use_channel_scales',[False,True])
def test_auto_int8_validates_actual_same_shaped_weights_before_reusing_timings(monkeypatch,use_channel_scales):
  from tinygrad.llm.kernels import rockchip
  from tinygrad.llm.kernels.rockchip_int8 import RockchipInt8Linear
  Tensor.manual_seed(31)
  first,second = nn.Linear(512,32,bias=False),nn.Linear(512,32,bias=False)
  first.weight = Tensor.ones(32,512,device='CPU').realize()
  second.weight = first.weight if use_channel_scales else Tensor.cat(
    Tensor.full((32,1),100.,device='CPU'),Tensor.ones(32,511,device='CPU'),dim=1).realize()
  model = SimpleNamespace(layers=[first,second])
  calls = {}
  def quantized(x):
    scale = (x.abs().max(-1,keepdim=True)/127).maximum(1e-8)
    return (x/scale).round().clip(-127,127)*scale
  def prepare(self):
    self.packed = [quantized(self.weight if self.input_scale is None else self.weight*self.input_scale).realize()]
  def npu(self,x):
    calls[id(self)] = calls.get(id(self),0)+1
    return quantized(x if self.input_scale is None else x/self.input_scale) @ self.packed[0].T
  monkeypatch.setattr(RockchipInt8Linear,'prepare',prepare)
  monkeypatch.setattr(RockchipInt8Linear,'npu',npu)
  medians = iter([1.,.1])
  monkeypatch.setattr(rockchip.statistics,'median',lambda samples:next(medians))
  scales = {'layers.0':Tensor.ones(512,device='CPU'),
            'layers.1':Tensor.cat(Tensor([100.],device='CPU'),Tensor.ones(511,device='CPU'))} if use_channel_scales else None
  result = configure(model,prefill='auto',prefill_rows=64,precision='int8',input_scales=scales)
  good,bad = result['decisions']
  assert good['normalized_rmse']<=.03<bad['normalized_rmse']
  assert good['selected'] and not bad['selected']
  assert [layer.prefill for layer in model.layers]==['on','off']
  assert not good['timing_reused'] and bad['timing_reused']
  assert good['original_ms']==bad['original_ms']==1000.
  assert good['npu_ms']==bad['npu_ms']==100.
  # A second median() call would exhaust medians; only numerical qualification
  # invokes the second layer, once. The first layer also runs its timed warmups.
  assert calls[id(model.layers[0])]>1 and calls[id(model.layers[1])]==1


def test_auto_fp16_also_validates_each_layer_when_shape_timing_is_reused(monkeypatch):
  from tinygrad.llm.kernels import rockchip
  first,second = nn.Linear(32,32,bias=False),nn.Linear(32,32,bias=False)
  first.weight = Tensor(np.eye(32,dtype=np.float32),device='CPU').realize()
  second.weight = (first.weight*2).realize()
  model = SimpleNamespace(layers=[first,second])
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  calls = {}
  def npu(self,x):
    calls[id(self)] = calls.get(id(self),0)+1
    return self.source(x)+(1 if self.source is second else 0)
  monkeypatch.setattr(RockchipLinear,'npu',npu)
  medians = iter([1.,.1])
  monkeypatch.setattr(rockchip.statistics,'median',lambda samples:next(medians))
  result = configure(model,prefill='auto',prefill_rows=16)
  good,bad = result['decisions']
  assert good['valid'] and good['selected'] and not good['timing_reused']
  assert not bad['valid'] and not bad['selected'] and bad['timing_reused']
  assert model.layers[0].prefill=='on' and model.layers[1].prefill=='off'
  assert calls[id(model.layers[1])]==1


def test_placement_cache_distinguishes_prefill_chunk_from_equal_shaped_output(monkeypatch):
  # K/V and the output share a shape, but only the output runs on one row.
  # Replace hardware work; the test checks scheduling decisions, not NPU speed.
  Tensor.manual_seed(31)
  model = Transformer(TransformerConfig(1,64,128,4,2,1e-5,32,16,10000.,16,16,max_context=128))
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  monkeypatch.setattr(RockchipLinear,'npu',lambda self,x:self.source(x))
  result = configure(model,prefill='auto',decode='off',prefill_rows=64)
  rows = {item['rows'] for item in result['decisions'] if item['phase']=='prefill' and item['shape']==[32,64]}
  assert rows=={1,64}


def test_three_kv_long_model_activates_decode_pool_and_short_model_keeps_original_path(monkeypatch):
  from types import SimpleNamespace
  from tinygrad.llm.kernels import rockchip_pool
  pool = SimpleNamespace(address=1234,active=False)
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:pool)
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  for context in (512,4096):
    model = Transformer(TransformerConfig(1,144,288,9,3,1e-5,32,16,10000.,16,16,max_context=context))
    result = configure(model,decode='on',cpu_attention=True,cpu_pool=True)
    assert result['cpu_pool']==(context>=4096)
    assert model.blk[0].decode_attention.keywords['pool'] is (pool if context>=4096 else None)


def test_recurrent_pool_is_created_for_small_context_before_cpu_recurrent_setup_and_prefill_uses_it(monkeypatch):
  from tinygrad.llm.kernels import rockchip_pool
  pool=_FakePool()
  creations=[]
  prefill_observations=[]
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:(creations.append('created') or pool))
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  model=_cpu_gdn_model(context=512,n_kv_heads=1,num_v_heads=4)

  class PreparedPrefill:
    def __call__(self,*args,**kwargs):
      prefill_observations.append(pool.active)
      return 'fake-prefill'

  model.prefill_jit=PreparedPrefill()
  result=configure(model,cpu_pool=True,cpu_recurrent_scan=True)
  hook=model.blk[0].gated_delta_scan
  assert creations==['created']
  assert result['cpu_recurrent_scan'] and result['cpu_recurrent_blocks']==1 and result['cpu_recurrent_pool']
  assert result['cpu_pool'] is True and hook.func.__name__=='gated_delta_scan' and hook.keywords['pool'] is pool
  assert model.prefill_jit()=='fake-prefill'
  assert prefill_observations==[True] and pool.prefill_entries==1 and not pool.active


def test_recurrent_pool_start_failure_keeps_serial_scan_and_reports_no_pool(monkeypatch):
  from tinygrad.llm.kernels import rockchip_pool, rockchip_recurrent
  creations=[]
  def fail_start():
    creations.append('attempted')
    raise OSError('fake pool startup failure')
  monkeypatch.setattr(rockchip_pool,'attention_pool',fail_start)
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  model=_cpu_gdn_model(context=512,n_kv_heads=1,num_v_heads=4)
  result=configure(model,cpu_pool=True,cpu_recurrent_scan=True)
  assert creations==['attempted']
  assert result['cpu_recurrent_scan'] and result['cpu_recurrent_blocks']==1 and result['cpu_recurrent_pool'] is False
  assert result['cpu_pool'] is False
  assert model.blk[0].gated_delta_scan is rockchip_recurrent.gated_delta_scan


def test_default_recurrent_scan_does_not_start_pool_for_two_kv_heads_at_context512(monkeypatch):
  from tinygrad.llm.kernels import rockchip_pool
  creations=[]
  monkeypatch.setattr(rockchip_pool,'attention_pool',lambda:(creations.append('unexpected') or _FakePool()))
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  model=_cpu_gdn_model(context=512,n_kv_heads=2,num_v_heads=2)
  result=configure(model,cpu_pool=True,cpu_attention=True)
  assert creations==[]
  assert result['cpu_pool'] is False and result['cpu_recurrent_scan'] is False
  assert result['cpu_recurrent_pool'] is False


def test_cpu_auto_calibration_uses_runtime_wake_helper_and_symbolic_rows(monkeypatch):
  from tinygrad.llm.kernels import rockchip_idle
  class Helper:
    active,entries = False,0
    def __enter__(self):
      self.active = True
      self.entries += 1
      return self
    def __exit__(self,*args): self.active = False
  helper = Helper()
  monkeypatch.setattr(rockchip_idle,'idle_helper',lambda:helper)
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  seen = []
  def npu(self,x):
    assert helper.active
    seen.append((x.shape[1],x.max_shape[1]))
    return self.source(x)
  monkeypatch.setattr(RockchipLinear,'npu',npu)
  model = Transformer(TransformerConfig(1,64,128,4,2,1e-5,32,16,10000.,16,16,max_context=128))
  result = configure(model,prefill='auto',prefill_rows=64,native_replay=True,wake_helper=True)
  assert helper.entries>0 and not helper.active
  assert any(isinstance(count,UOp) and maximum==64 for count,maximum in seen)
  assert result['wake_helper'] and all(d['wake_helper'] for d in result['decisions'])


def test_int8_score_dot_is_opt_in_and_wires_prefill_and_decode_together(monkeypatch):
  import pytest
  from tinygrad.llm.kernels.rockchip_dot_attention import Int8CPUAttention, supports_dotprod
  if not supports_dotprod(): pytest.skip('AArch64 SDOT required')
  monkeypatch.setattr(RockchipLinear,'prepare',lambda self:None)
  model = Transformer(TransformerConfig(1,576,1152,9,3,1e-5,32,64,100000.,64,64,max_context=4096))
  with pytest.raises(ValueError,match='requires packed'): configure(model,cpu_int8_qk=True)
  with pytest.raises(ValueError,match='Choose one'):
    configure(model,cpu_attention=True,cpu_fast_softmax=True,cpu_packed_attention=True,cpu_fp16_qk=True,cpu_int8_qk=True)
  result = configure(model,decode='on',cpu_attention=True,cpu_pool=True,cpu_fast_softmax=True,
                     cpu_packed_attention=True,cpu_int8_qk=True)
  attention = model.blk[0].prefill_attention
  assert result['cpu_int8_qk'] and not result['cpu_fp16_qk'] and result['packed_attention_blocks']==1
  assert isinstance(attention,Int8CPUAttention) and model.blk[0].decode_attention.func.__self__ is attention
