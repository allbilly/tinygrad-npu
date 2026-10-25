"""Prefix-cache equivalence and avoided projection work for the trained draft."""
import os, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2 import DFlash2
from tinygrad.llm.model import TargetOutput
from test.models.test_dflash2 import fixture


def cached_fixture(seed=473, *, cache=True):
  base,c,w,embedding,head,rng = fixture(seed)
  draft = DFlash2(c,base.weights,base.token_embedding,base.lm_head,base.verifier_norm,cache_prefix_kv=cache)
  return draft,c,rng


def output(features):
  tensors = tuple(Tensor(x,device='CPU').realize() for x in np.split(features,2,axis=-1))
  return TargetOutput(Tensor.zeros(1,features.shape[1],16,device='CPU'),tensors[-1],tensors)


def forwards(draft,features,position):
  noise = Tensor([[3,15,15,15]],device='CPU',dtype=dtypes.int32)
  hidden = draft.project_features(Tensor(features,device='CPU')).realize()
  raw = draft._forward_projected(noise,hidden,position=position).realize()
  cached = draft._forward_projected(noise,draft._target_hidden,position=position,prefix_kv=draft._prefix_kv).realize()
  return raw,cached


class TestDFlash2Cache(unittest.TestCase):
  def assert_same_forward(self,draft,features):
    raw,cached = forwards(draft,features,features.shape[1])
    np.testing.assert_allclose(cached.logits.numpy(),raw.logits.numpy(),atol=2e-6,rtol=3e-5)
    np.testing.assert_allclose(cached.hidden.numpy(),raw.hidden.numpy(),atol=2e-6,rtol=3e-5)
    np.testing.assert_array_equal(cached.logits.numpy().argmax(-1),raw.logits.numpy().argmax(-1))

  def test_cached_forward_matches_raw_with_long_sliding_prefix(self):
    draft,c,rng = cached_fixture()
    features = rng.normal(0,.4,(1,129,16)).astype(np.float32)
    draft.observe(output(features[:,:128]),0)
    draft.observe(output(features[:,128:]),128)
    self.assert_same_forward(draft,features)
    self.assertEqual((draft._prefix_kv[0].start_pos,draft._prefix_kv[0].end_pos),(127,129))
    self.assertEqual(draft._prefix_kv[0].key.shape,(1,1,2,4))
    self.assertEqual((draft._prefix_kv[1].start_pos,draft._prefix_kv[1].end_pos),(0,129))
    self.assertEqual(draft._prefix_kv[1].key.shape,(1,1,129,4))
    c.sliding_window_non_causal=False
    self.assert_same_forward(draft,features)

  def test_only_new_target_rows_are_projected_and_proposals_reuse_prefix_kv(self):
    draft,_c,rng = cached_fixture(474)
    calls=[]
    linear=draft._linear
    def tracked(x,name):
      calls.append((name,x.shape[1]))
      return linear(x,name)
    draft._linear=tracked
    features=rng.normal(0,.4,(1,7,16)).astype(np.float32)
    draft.observe(output(features[:,:6]),0)
    revision = draft.cache_revision
    self.assertEqual(draft.cached_prefix_length,6)
    initial=[(name,rows) for name,rows in calls if name.endswith(('.k_proj','.v_proj'))]
    self.assertEqual(initial,[('layers.0.self_attn.k_proj',2),('layers.0.self_attn.v_proj',2),
                              ('layers.1.self_attn.k_proj',6),('layers.1.self_attn.v_proj',6)])
    calls.clear()
    draft.observe(output(features[:,6:]),6)
    self.assertGreater(draft.cache_revision,revision)
    self.assertEqual(draft.cached_prefix_length,7)
    revision = draft.cache_revision
    appended=[(name,rows) for name,rows in calls if name.endswith(('.k_proj','.v_proj'))]
    self.assertEqual(appended,[('layers.0.self_attn.k_proj',1),('layers.0.self_attn.v_proj',1),
                               ('layers.1.self_attn.k_proj',1),('layers.1.self_attn.v_proj',1)])
    calls.clear()
    noise=Tensor([[3,15,15,15]],device='CPU',dtype=dtypes.int32)
    draft._forward_projected(noise,draft._target_hidden,position=7,prefix_kv=draft._prefix_kv).realize()
    self.assertEqual(draft.cache_revision,revision)
    kv_calls=[(name,rows) for name,rows in calls if name.endswith(('.k_proj','.v_proj'))]
    self.assertEqual(kv_calls,[('layers.0.self_attn.k_proj',4),('layers.0.self_attn.v_proj',4),
                              ('layers.1.self_attn.k_proj',4),('layers.1.self_attn.v_proj',4)])
    calls.clear()
    draft._forward_projected(noise,draft._target_hidden,position=7).realize()
    raw_calls=[(name,rows) for name,rows in calls if name.endswith(('.k_proj','.v_proj'))]
    self.assertEqual(sum(rows for _name,rows in raw_calls),4*(7+4))
    self.assertEqual(sum(rows for _name,rows in kv_calls),4*4)

  def test_suffix_replacement_deep_rollback_and_empty_truncation(self):
    draft,_c,rng = cached_fixture(475)
    features=np.empty((1,0,16),np.float32)
    for position,rows in ((0,8),(7,3),(2,1),(3,4),(4,0)):
      new=rng.normal(0,.4,(1,rows,16)).astype(np.float32)
      features=np.concatenate((features[:,:position],new),axis=1)
      draft.observe(output(new),position)
      self.assert_same_forward(draft,features)
      for index,cache in enumerate(draft._prefix_kv):
        self.assertEqual(cache.end_pos,features.shape[1])
        self.assertEqual(cache.start_pos,max(0,features.shape[1]-2) if index==0 else 0)
    draft.observe(output(np.empty((1,0,16),np.float32)),0)
    self.assertIsNone(draft._target_hidden)
    self.assertEqual(draft._prefix_kv,(None,None))

  def test_failed_observation_preserves_matching_features_and_kv(self):
    draft,_c,rng = cached_fixture(476)
    features=rng.normal(0,.4,(1,5,16)).astype(np.float32)
    draft.observe(output(features),0)
    saved_hidden,saved_kv=draft._target_hidden,draft._prefix_kv
    revision = draft.cache_revision
    build=draft._make_prefix_kv
    def fail_layer(hidden,start_pos,layer):
      if layer==1:raise RuntimeError('injected projection failure')
      return build(hidden,start_pos,layer)
    draft._make_prefix_kv=fail_layer
    with self.assertRaisesRegex(RuntimeError,'injected projection'):
      draft.observe(output(rng.normal(0,.4,(1,1,16)).astype(np.float32)),5)
    self.assertGreater(draft.cache_revision,revision)
    self.assertEqual(draft.cached_prefix_length,5)
    self.assertIs(draft._target_hidden,saved_hidden)
    self.assertIs(draft._prefix_kv,saved_kv)
    self.assert_same_forward(draft,features)

  def test_disabled_cache_preserves_the_original_arithmetic_path(self):
    draft,_c,rng = cached_fixture(477,cache=False)
    features=rng.normal(0,.4,(1,7,16)).astype(np.float32)
    draft.observe(output(features),0)
    self.assertEqual(draft._prefix_kv,(None,None))
    self.assert_same_forward(draft,features)
    with self.assertRaisesRegex(ValueError,'Mali OpenCL'):draft.prepare_mali()

  def test_prepared_projection_adapter_preserves_activation_dtype(self):
    draft,_c,_rng = cached_fixture(478)
    weight=draft.weights['fc.weight']
    activations=Tensor.ones(1,2,16,device='CPU',dtype=dtypes.float16)
    received=[]
    def prepared(x):
      received.append(x.dtype)
      return x.linear(weight.transpose())
    draft._prepared_linear['fc']=prepared
    actual=draft._linear(activations,'fc')
    self.assertEqual(actual.dtype,dtypes.float16)
    self.assertEqual(received,[dtypes.float32])
    expected=activations.float().linear(weight.transpose()).cast(dtypes.float16)
    np.testing.assert_array_equal(actual.numpy(),expected.numpy())


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware qualification')
class TestDFlash2Mali(unittest.TestCase):
  def fixture(self,seed):
    _base,c,weights,embedding,head,rng=fixture(seed)
    tensors={name:Tensor(value.astype(np.float16),device='CL').realize() for name,value in weights.items()}
    emb=Tensor(embedding.astype(np.float16),device='CL').realize()
    output_head=Tensor(head.astype(np.float16),device='CL').realize()
    def embedding_fn(ids):return emb[ids]
    def head_fn(hidden):return hidden@output_head.T
    raw=DFlash2(c,tensors,embedding_fn,head_fn,lambda hidden:hidden,cache_prefix_kv=False)
    prepared=DFlash2(c,tensors,embedding_fn,head_fn,lambda hidden:hidden,cache_prefix_kv=True)
    metadata=prepared.prepare_mali(pack_weights=False)
    self.assertEqual(metadata,{'linear_layers':19,'packed_weights':False,'prepared_shared_head':False,'packed_fc':False,'packed_bytes':0})
    for name,layer in prepared._prepared_linear.items():
      self.assertIs(layer.weight,tensors[name+'.weight'])
      self.assertIsNone(layer.packed_weight)
    return raw,prepared,rng

  def test_fp16_prepared_fc_attention_and_convolution_projections(self):
    raw,prepared,rng=self.fixture(481)
    for name in ('fc','layers.0.self_attn.k_proj','layers.0.attention_conv.kernel_projection'):
      weight=raw.weights[name+'.weight']
      values=rng.normal(0,.4,(1,17,weight.shape[1])).astype(np.float16)
      x=Tensor(values,device='CL').realize()
      expected=values.astype(np.float32)@weight.float().numpy().T
      ordinary=raw._linear(x,name).numpy()
      actual=prepared._linear(x,name)
      self.assertEqual(actual.dtype,dtypes.float16)
      np.testing.assert_allclose(ordinary,expected,atol=2e-3,rtol=2e-3)
      np.testing.assert_allclose(actual.numpy(),expected,atol=2e-3,rtol=2e-3)
      np.testing.assert_allclose(actual.numpy(),ordinary,atol=2e-3,rtol=2e-3)

  def test_fp16_cached_prepared_forward_matches_raw_prefix_and_deep_rollback(self):
    raw,prepared,rng=self.fixture(482)
    features=rng.normal(0,.4,(1,19,16)).astype(np.float16)
    prepared.observe(output(features[:,:17]),0)
    prepared.observe(output(features[:,17:]),17)
    noise=Tensor([[3,15,15,15]],device='CL',dtype=dtypes.int32).realize()
    for rollback in (False,True):
      if rollback:
        replacement=rng.normal(0,.4,(1,1,16)).astype(np.float16)
        features=np.concatenate((features[:,:5],replacement),axis=1)
        prepared.observe(output(replacement),5)
      saved=[(cache.key.numpy().copy(),cache.value.numpy().copy()) for cache in prepared._prefix_kv]
      ordinary=raw(noise,Tensor(features,device='CL').realize(),position=features.shape[1]).realize()
      actual=prepared._forward_projected(noise,prepared._target_hidden,position=features.shape[1],prefix_kv=prepared._prefix_kv).realize()
      np.testing.assert_allclose(actual.logits.numpy(),ordinary.logits.numpy(),atol=3e-3,rtol=4e-3)
      np.testing.assert_allclose(actual.hidden.numpy(),ordinary.hidden.numpy(),atol=3e-3,rtol=4e-3)
      np.testing.assert_array_equal(actual.logits.numpy().argmax(-1),ordinary.logits.numpy().argmax(-1))
      for cache,(key,value) in zip(prepared._prefix_kv,saved):
        np.testing.assert_array_equal(cache.key.numpy(),key)
        np.testing.assert_array_equal(cache.value.numpy(),value)

  def test_fp16_borrowed_prepared_head_preserves_dtype_sharing_and_greedy_tokens(self):
    from tinygrad import nn
    from tinygrad.llm.kernels.mali import MaliLinear
    _raw,draft,rng=self.fixture(483)
    source=nn.Linear(8,16,bias=False)
    weights=rng.normal(0,.2,(16,8)).astype(np.float16)
    source.weight.replace(Tensor(weights,device='CL').realize())
    shared_head=MaliLinear(source,prepare_gemm=False)
    draft.lm_head=shared_head
    original_weight=shared_head.weight
    metadata=draft.prepare_mali(pack_weights=False,prepare_shared_head=True)
    self.assertTrue(metadata['prepared_shared_head'])
    values=rng.normal(0,.4,(1,4,8)).astype(np.float16)
    hidden=Tensor(values,device='CL').realize()
    expected=(values.astype(np.float32)@weights.astype(np.float32).T).astype(np.float16)
    actual=draft._project_head(hidden)
    self.assertEqual(actual.dtype,dtypes.float16)
    np.testing.assert_allclose(actual.numpy(),expected,atol=2e-3,rtol=2e-3)
    np.testing.assert_array_equal(actual.numpy().argmax(-1),expected.argmax(-1))
    self.assertIs(draft.lm_head,shared_head)
    self.assertIs(shared_head.weight,original_weight)
    self.assertIs(source.weight,original_weight)
    self.assertIsNone(shared_head.packed_weight)
    np.testing.assert_array_equal(shared_head.weight.numpy(),weights)
    metadata=draft.prepare_mali(pack_weights=False,prepare_shared_head=False)
    self.assertFalse(metadata['prepared_shared_head'])
    ordinary=draft._project_head(hidden)
    np.testing.assert_array_equal(ordinary.numpy(),shared_head(hidden).numpy())
    self.assertIs(draft.lm_head,shared_head)


if __name__=='__main__':unittest.main()
