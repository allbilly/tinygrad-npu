"""Combined INT8 dense/grouped MLA generation versus CPU integer projection math."""
import itertools,os,unittest
import numpy as np
from tinygrad import Tensor,dtypes,nn
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.rockchip import configure
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.model import Transformer,TransformerConfig


class CPUReference:
  def __init__(self,key,value,residual=False,projections='both'):
    self.residual=residual
    self.projections=projections
    self._sources=lambda:(key,value)
    pairs=[]
    for weight in (key,value):
      groups,columns,depth=weight.shape
      payload,scale=quantize(weight.reshape(groups*columns,depth),depth)
      codes=payload[round_up(groups*columns*4,64):].bitcast(dtypes.int8).reshape(groups,columns,depth).contiguous().realize()
      scales=scale.reshape(groups,columns).contiguous().realize()
      if residual:
        from tinygrad.llm.kernels.rockchip_grouped_int8 import _remainder
        remainder=_remainder(weight,codes,scales)
        extra,extra_scales=quantize(remainder.reshape(groups*columns,depth),depth)
        extra_codes=extra[round_up(groups*columns*4,64):].bitcast(dtypes.int8).reshape(groups,columns,depth)
        codes=codes.cat(codes,extra_codes,dim=0).contiguous().realize()
        scales=scales.cat(scales,extra_scales.reshape(groups,columns),dim=0).contiguous().realize()
      pairs.append((codes,scales))
    self._weights=lambda:tuple(pairs)
    self.phase=None

  def run(self,x,key):
    if self.projections==('value' if key else 'query'):return x@self._sources()[0 if key else 1].transpose(-1,-2)
    weight,scales=self._weights()[0 if key else 1]
    groups,columns,depth=x.shape[1],weight.shape[1],weight.shape[2]
    capacity=x.max_shape[2]
    dense=x.pad_to((1,groups,capacity,depth)).reshape(groups*capacity,depth)
    payload,activation_scales=quantize(dense,depth)
    code=payload[round_up(groups*capacity*4,64):].bitcast(dtypes.int8).reshape(groups,capacity,depth)
    if self.residual:
      from tinygrad.llm.kernels.rockchip_grouped_int8 import _remainder
      remainder=_remainder(dense.reshape(groups,capacity,depth),code,activation_scales)
      extra,extra_scales=quantize(remainder.reshape(groups*capacity,depth),depth)
      extra_codes=extra[round_up(groups*capacity*4,64):].bitcast(dtypes.int8).reshape(groups,capacity,depth)
      code=code.cat(extra_codes,code,dim=0)
      activation_scales=activation_scales.cat(extra_scales,activation_scales)
    count=groups*(3 if self.residual else 1)
    integer=code.int()@weight.int().transpose(-1,-2)
    result=(integer.float()*activation_scales.reshape(count,capacity,1))*scales.reshape(count,1,columns)
    if self.residual:
      parts=result.reshape(3,groups,capacity,columns)
      result=(parts[0]+parts[1])+parts[2]
    return result.reshape(1,groups,capacity,columns)[:,:,:x.shape[2]]

  def query(self,x):return self.run(x,True)
  def value(self,x):return self.run(x,False)


@unittest.skipUnless(os.environ.get('ROCKCHIP_TEST')=='1','explicit Rockchip hardware test')
class TestGroupedInt8Model(unittest.TestCase):
  def test_full_dense_and_grouped_capture_generation_reset(self):self._check()

  def test_full_residual_capture_generation_reset(self):self._check(residual=True)

  def test_query_only_residual_capture_generation_reset(self):self._check(residual=True,projections='query')

  def _check(self,residual=False,projections='both'):
    config=TransformerConfig(2,320,640,40,40,1e-5,64,96,10000.,32,8,max_context=512,q_lora_rank=32,kv_lora_rank=256)
    reference,target=Transformer(config),Transformer(config)
    rng=np.random.default_rng(271109)
    weights={}
    for name,param in nn.state.get_state_dict(reference).items():
      value=rng.normal(0,.02,param.shape).astype(np.float16)
      if 'norm' in name:value+=1
      weights[name]=Tensor(value,device='CPU').realize()
    for model in (reference,target):
      nn.state.load_state_dict(model,weights)
      options=dict(prefill='on',decode='on',tile_columns=24576,precision='int8',native_replay=True,
        fuse_projections=True,share_projection_weights=True,fused_ffn=True,split_down=True,fused_qkv=True,
        cpu_norm=True,cpu_attention=True,cpu_pool=True,cpu_prefill_pool=True,cpu_fast_softmax=True,
        cache_cleaning=True,wake_helper=True,cpu_mla_weights=True,cpu_mla_attention=True)
      if model is target:options.update(npu_mla_prefill=True,npu_mla_decode=True,
        npu_mla_precision='int8_residual' if residual else 'int8',npu_mla_projections=projections)
      placement=configure(model,**options)
      self.assertEqual(placement['layers'],15)
      if model is target:
        self.assertEqual(placement['npu_mla_projection_blocks'],2)
        self.assertEqual(placement['npu_mla_projection_precision'],'int8_residual' if residual else 'int8')
        self.assertEqual(placement['npu_mla_projection_selection'],projections)
      else:
        for block in model.blk:block.npu_mla_projections=CPUReference(block.attn_k_b['weight'],block.attn_v_b['weight'],
          residual=residual,projections=projections)
    for request in range(4):
      prefix=[(i+request)%63+1 for i in range(128)]
      outputs=[]
      for model in (reference,target):
        model._cached_tokens=[]
        outputs.append(list(itertools.islice(model.generate(prefix.copy(),chunk_size=128,temperature=0),8)))
      self.assertEqual(outputs[0],outputs[1])
    # Native decode is required for the target; a Python fallback cannot pass.
    wrapper=target.greedy_rollout_jit
    raw_jit=next(cell.cell_contents for cell in wrapper.__closure__ if hasattr(cell.cell_contents,'captured'))
    self.assertGreater(raw_jit.captured.native_runs,0)
    self.assertIsNone(raw_jit.captured.reason)


if __name__=='__main__':unittest.main()
