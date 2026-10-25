"""Independent recurrence, mutable-state and full GatedDeltaNet checks on Mali."""
import os, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.llm.kernels.mali_recurrent import gated_delta_scan
from tinygrad.uop.ops import Ops


def reference(q,k,v,beta,alpha,state):
  outputs = []
  for index in range(q.shape[2]):
    decay = alpha[:,:,index]
    if decay.ndim==2:decay=decay[:,:,None]
    decayed = state*decay[:,:,None,:]
    delta = (v[:,:,index]-(decayed*k[:,:,index,None,:]).sum(-1))*beta[:,:,index,None]
    state = decayed+delta[:,:,:,None]*k[:,:,index,None,:]
    outputs.append((state*q[:,:,index,None,:]).sum(-1))
  return np.stack(outputs,axis=2),state


def operands(rng,batch,heads,rows,key_dim,value_dim,alpha_dim):
  q,k = (rng.normal(0,.2,(batch,heads,rows,key_dim)).astype(np.float32) for _ in range(2))
  q /= np.sqrt((q*q).sum(-1,keepdims=True)+1e-6)*np.float32(key_dim**.5)
  k /= np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
  v = rng.normal(0,.2,(batch,heads,rows,value_dim)).astype(np.float32)
  beta = rng.uniform(.1,.9,(batch,heads,rows)).astype(np.float32)
  alpha = rng.uniform(.8,.99,(batch,heads,rows,alpha_dim)).astype(np.float32)
  return q,k,v,beta,alpha


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliRecurrent(unittest.TestCase):
  def test_outputs_state_and_reset_match_independent_recurrence(self):
    rng = np.random.default_rng(3510)
    for batch,heads,rows,key_dim,value_dim,alpha_dim in ((1,2,7,8,5,1),(2,3,17,64,13,1),(1,16,128,128,128,1),
                                                        (1,2,1,128,128,1),(1,2,9,128,16,128),(1,2,4,256,8,256)):
      values = operands(rng,batch,heads,rows,key_dim,value_dim,alpha_dim)
      initial = rng.normal(0,.03,(batch,heads,value_dim,key_dim)).astype(np.float32)
      state = Tensor(initial.copy(),device='CL').realize()
      tensors = tuple(Tensor(x,device='CL').realize() for x in values)
      actual = gated_delta_scan(*tensors,state,5,rows).numpy()
      expected,final = reference(*values,initial)
      np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-4)
      np.testing.assert_allclose(state.numpy(),final,atol=2e-6,rtol=2e-4)
      state.assign(Tensor.full(initial.shape,float('nan'),device='CL')).realize()
      actual = gated_delta_scan(*tensors,state,0,rows).numpy()
      expected,final = reference(*values,np.zeros_like(initial))
      np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-4)
      np.testing.assert_allclose(state.numpy(),final,atol=2e-6,rtol=2e-4)

  def test_symbolic_rows_positions_padding_and_scratch_release(self):
    rng = np.random.default_rng(3511)
    batch,heads,capacity,key_dim,value_dim = 1,3,17,64,11
    state = Tensor.zeros(batch,heads,value_dim,key_dim,device='CL').realize()
    expected_state = np.zeros(state.shape,np.float32)
    run = TinyJit(lambda q,k,v,beta,alpha,p,r:gated_delta_scan(q,k,v,beta,alpha,state,p,r).realize())
    for index,(count,position) in enumerate(((17,0),(1,17),(13,18),(7,0),(16,7),(17,0),(3,17),(9,20))):
      values = operands(rng,batch,heads,capacity,key_dim,value_dim,1)
      valid = tuple(x[:,:,:count].copy() for x in values)
      for x in values:x[:,:,count:]=np.nan
      if position==0:expected_state.fill(0)
      expected,expected_state = reference(*valid,expected_state)
      actual = run(*(Tensor(x,device='CL').realize() for x in values),
                   UOp.variable('scan_position',0,512).bind(position),UOp.variable('scan_rows',1,capacity).bind(count)).numpy()
      np.testing.assert_allclose(actual[:,:,:count],expected,atol=2e-6,rtol=2e-4)
      self.assertFalse(actual[:,:,count:].any())
      np.testing.assert_allclose(state.numpy(),expected_state,atol=2e-6,rtol=2e-4)
      if index==4:run.captured.free_intermediates()
    self.assertEqual(sum(call.body.op is Ops.PROGRAM and call.body.src[0].arg.function_name=='mali_gated_delta_scan'
                         for call in run.captured.linear.src),1)

  def test_rejects_permuted_state_and_invalid_row_bounds(self):
    values = tuple(Tensor(x,device='CL').realize() for x in operands(np.random.default_rng(3512),1,2,7,8,5,1))
    state = Tensor.zeros(1,2,5,8,device='CL').realize()
    permuted = Tensor.zeros(1,2,8,5,device='CL').realize().permute(0,1,3,2)
    with self.assertRaises(ValueError):gated_delta_scan(*values,permuted,0,7)
    for start,rows in ((0,0),(0,8),(-1,7),(0,UOp.variable('invalid_scan_rows',0,7).bind(0))):
      with self.assertRaises(ValueError):gated_delta_scan(*values,state,start,rows)
    self.assertFalse(state.numpy().any())

  def test_full_block_and_bulk_prefill_policy(self):
    from test.models.test_gated_delta_layout import TestGatedDeltaLayout
    from tinygrad.llm.model import GatedDeltaNetBlock, Transformer
    from tinygrad.llm.kernels.mali import MaliLinear, configure
    cpu,weights,ssm = TestGatedDeltaLayout().fixture(4)
    model = Transformer(replace(cpu.config,ssm_layers=(True,)))
    block = GatedDeltaNetBlock(model.config,ssm)
    for name,p in nn.state.get_state_dict(block).items():
      p.replace(nn.state.get_state_dict(cpu)[name].to('CL').realize())
    model.blk[0] = block
    for parameter in nn.state.get_parameters(model):parameter.to_('CL')
    self.assertEqual(model.prefill_chunk_size(16),1)
    block.ffn_gate = MaliLinear(block.ffn_gate)
    self.assertIsNotNone(block.ffn_gate.packed_weight)
    for _ in range(2):
      metadata = configure(model,recurrent_scan=True)
      self.assertEqual(metadata['fused_gated_delta_blocks'],1)
      self.assertFalse(metadata['recurrent_pack_weights'])
      for name in ('attn_qkv','attn_gate','ssm_alpha','ssm_beta','ssm_out','ffn_gate','ffn_up','ffn_down'):
        self.assertIsNone(getattr(block,name).packed_weight)
      self.assertEqual(model.prefill_chunk_size(16),16)
      self.assertEqual(model.prefill_chunk_size(512),256)
    values = (np.random.default_rng(35).integers(-8,9,(1,9,32))/16).astype(np.float32)
    block._init_state(Tensor(values[:,:1],device='CL'))
    start = 0
    for end in (6,7,9):
      actual = block._attention(Tensor(values[:,start:end],device='CL'),start).numpy()
      expected = TestGatedDeltaLayout().reference(values[:,:end],weights,ssm,block.config.norm_eps)[:,start:end]
      np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-3)
      start = end
    configure(model,pack_weights=False,recurrent_scan=False)
    self.assertEqual(model.prefill_chunk_size(16),1)

  def test_fp16_hybrid_model_bulk_logits_match_single_token_execution(self):
    from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig
    from tinygrad.llm.kernels.mali import configure
    ssm = SSMConfig(3,8,2,4,32)
    config = TransformerConfig(2,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=24,ssm=ssm,ssm_layers=(True,False))
    ordinary,bulk = Transformer(config),Transformer(config)
    rng = np.random.default_rng(3513)
    for name,p in nn.state.get_state_dict(ordinary).items():
      values = rng.normal(0,.12,p.shape).astype(np.float32)
      if 'norm.weight' in name:values+=1
      if name.endswith('ssm_a'):values=-rng.uniform(.5,2,p.shape).astype(np.float32)
      p.replace(Tensor(values.astype(np.float16),device='CL').realize())
    for parameter in nn.state.get_parameters(bulk):parameter.to_('CL')
    nn.state.load_state_dict(bulk,{name:value.clone().realize() for name,value in nn.state.get_state_dict(ordinary).items()},verbose=False)
    configure(ordinary,pack_weights=False,recurrent_scan=False)
    configure(bulk,pack_weights=False,recurrent_scan=True)
    tokens = [2,5,11,17,7,31,29,19,3]
    start = 0
    for end in (1,6,7,9):
      for index in range(start,end):expected=ordinary.logits(Tensor([[tokens[index]]],device='CL'),index).numpy()
      actual = bulk.logits(Tensor([tokens[start:end]],device='CL'),start).numpy()
      np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-3)
      np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
      start=end
    for index in range(3):expected=ordinary.logits(Tensor([[tokens[index]]],device='CL'),index).numpy()
    actual = bulk.logits(Tensor([tokens[:3]],device='CL'),0).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=3e-3)


if __name__=='__main__':unittest.main()
