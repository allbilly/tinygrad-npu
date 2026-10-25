"""Block verification outputs and feature/state contracts for trained speculation."""
import os, unittest
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig


def pair(config, device='CPU', half=False):
  ordinary,target = Transformer(config),Transformer(config)
  rng = np.random.default_rng(2110)
  for name,p in nn.state.get_state_dict(ordinary).items():
    values = rng.normal(0,.12,p.shape).astype(np.float32)
    if 'norm.weight' in name:values+=1
    if name.endswith('ssm_a'):values=-rng.uniform(.5,2,p.shape).astype(np.float32)
    p.replace(Tensor(values.astype(np.float16) if half else values,device=device).realize())
  for parameter in nn.state.get_parameters(target):parameter.to_(device)
  nn.state.load_state_dict(target,{k:v.clone().realize() for k,v in nn.state.get_state_dict(ordinary).items()},verbose=False)
  return ordinary,target


def scalar_reference(model, ids, start, layers):
  logits,hidden,features = [],[],{i:[] for i in layers}
  for position,token in enumerate(ids,start):
    x = model.token_embd(Tensor([[token]],device=model.token_embd.weight.device)).float()*model.config.embedding_scale
    for index,block in enumerate(model.blk):
      x = block(x,position).realize()
      if index in features:features[index].append(x.numpy().copy())
    hidden.append(x.numpy().copy())
    scores = model.output(model.output_norm(x[:,-1:])/model.config.logit_scale)[:,-1,:]
    if model.config.final_logit_softcap:
      scores=(scores/model.config.final_logit_softcap).tanh()*model.config.final_logit_softcap
    logits.append(scores.numpy().copy())
  return np.stack(logits,axis=1),np.concatenate(hidden,axis=1),tuple(np.concatenate(features[i],axis=1) for i in layers)


def config(hybrid=False):
  base = TransformerConfig(3,32,64,4,2,1e-5,64,8,10000.,8,8,max_context=32,
                           embedding_scale=1.5,logit_scale=1.25,final_logit_softcap=3.0)
  return replace(base,ssm=SSMConfig(3,8,2,4,32),ssm_layers=(True,False,True)) if hybrid else base


class TestSpeculativeTarget(unittest.TestCase):
  def test_full_logits_raw_features_order_and_ordinary_last_token_api(self):
    ordinary,target = pair(config())
    ids,layers = [2,5,11,7,3,19],(2,0,1)
    expected = scalar_reference(ordinary,ids,0,layers)
    tokens = Tensor([ids],device='CPU')
    actual = target.target_forward(tokens,0,capture_layers=layers)
    self.assertEqual(actual.logits.shape,(1,len(ids),64))
    for got,want in zip((actual.logits,actual.hidden,*actual.features),(expected[0],expected[1],*expected[2])):
      np.testing.assert_allclose(got.numpy(),want,atol=2e-5,rtol=3e-4)
    ordinary_last = target.logits(tokens,0).numpy()
    last = target.target_forward(tokens,0,capture_layers=(2,),last_only=True)
    self.assertEqual(last.logits.shape,(1,1,64))
    np.testing.assert_array_equal(last.logits.numpy()[:,0].view(np.uint32),ordinary_last.view(np.uint32))
    self.assertIs(last.hidden,last.features[0])
    self.assertEqual(target._cached_tokens,[])

  def test_hybrid_symbolic_jit_outputs_and_state_advance_once(self):
    ordinary,target = pair(config(True))
    layers=(0,2)
    run=TinyJit(lambda tokens,pos,rows:target.target_forward(tokens[:,:rows],pos,capture_layers=layers))
    cases=[([2,5,11,7],0),([3],4),([19,9,1],5),([8,3],0),([7,4,2,1],2),([9],6),([5,6,8],0)]
    for index,(ids,start) in enumerate(cases):
      expected=scalar_reference(ordinary,ids,start,layers)
      tokens=Tensor([ids+[0]*(4-len(ids))],device='CPU').realize()
      actual=run(tokens,UOp.variable('verify_pos',0,28).bind(start),UOp.variable('verify_rows',1,4).bind(len(ids)))
      for got,want in zip((actual.logits,actual.hidden,*actual.features),(expected[0],expected[1],*expected[2])):
        values=got.pad_to(got.max_shape).numpy()[:,:len(ids)]
        np.testing.assert_allclose(values,want,atol=3e-5,rtol=4e-4)
      for a,b in zip(target.blk,ordinary.blk):
        if hasattr(a,'recurrent_state'):
          before=a.recurrent_state.numpy().copy()
          np.testing.assert_allclose(before,b.recurrent_state.numpy(),atol=2e-6,rtol=2e-4)
          # Accessing any result again must not execute the stateful blocks again.
          actual.logits.pad_to(actual.logits.max_shape).numpy()
          actual.features[0].pad_to(actual.features[0].max_shape).numpy()
          np.testing.assert_array_equal(a.recurrent_state.numpy(),before)
          np.testing.assert_allclose(a.conv_state.numpy(),b.conv_state.numpy(),atol=3e-5,rtol=4e-4)
      if index==4:run.captured.free_intermediates()

  def test_invalid_capture_and_bounds_fail_before_cache_writes(self):
    _,target=pair(config())
    tokens=Tensor([[1,2]],device='CPU')
    for layers in ((0,0),(-1,),(3,),(True,)):
      with self.assertRaises(ValueError):target.target_forward(tokens,0,capture_layers=layers)
    for ids,pos in (([[1,2]],31),([[1]],-1),([[]],0)):
      with self.assertRaises(ValueError):target.target_forward(Tensor(ids,device='CPU'),pos)
    self.assertTrue(all(not hasattr(block,'cache_kv') for block in target.blk))

  def test_state_matches_ordinary_block_execution_bitwise(self):
    ordinary,target=pair(config(True))
    tokens=Tensor([[2,5,11,7]],device='CPU')
    expected=ordinary.logits(tokens,0).numpy()
    actual=target.target_forward(tokens,0,capture_layers=(0,2))
    np.testing.assert_array_equal(actual.logits.numpy()[:,-1].view(np.uint32),expected.view(np.uint32))
    for a,b in zip(target.blk,ordinary.blk):
      for name in ('cache_kv','conv_state','recurrent_state'):
        if hasattr(a,name):np.testing.assert_array_equal(getattr(a,name).numpy(),getattr(b,name).numpy())


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestSpeculativeTargetMali(unittest.TestCase):
  def test_fp16_hybrid_block_features_and_last_projection_match_scalar_gpu(self):
    from tinygrad.llm.kernels.mali import configure
    ordinary,target=pair(config(True),'CL',half=True)
    configure(ordinary,recurrent_scan=True,pack_weights=False)
    configure(target,recurrent_scan=True,pack_weights=False)
    ids,layers=[2,5,11,7,3,19],(0,2)
    expected=scalar_reference(ordinary,ids,0,layers)
    actual=target.target_forward(Tensor([ids],device='CL'),0,capture_layers=layers)
    for got,want in zip((actual.logits,actual.hidden,*actual.features),(expected[0],expected[1],*expected[2])):
      np.testing.assert_allclose(got.numpy(),want,atol=2e-3,rtol=3e-3)
    np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),expected[0].argmax(-1))
    before=target.blk[0].recurrent_state.numpy().copy()
    actual.features[0].numpy()
    actual.logits.numpy()
    actual.hidden.numpy()
    np.testing.assert_array_equal(target.blk[0].recurrent_state.numpy(),before)


if __name__=='__main__':unittest.main()
