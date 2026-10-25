import os, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, nn
from tinygrad.llm.kernels.mali import MaliLinear
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock


class TestPreparedMaliLinear(unittest.TestCase):
  def test_bounded_mask_matches_lower_right_causality_and_sliding_window(self):
    Tensor.manual_seed(39)
    for window in (0,13):
      block = TransformerBlock(TransformerConfig(1,64,128,4,2,1e-5,96,16,10000.,16,16,max_context=64,sliding_window=window))
      original = TinyJit(lambda x,start,rows:block(x[:,:rows],start).realize())
      bounded = TinyJit(lambda x,start,rows:block(x[:,:rows],start).realize())
      for start,rows in ((0,17),(8,31),(31,1),(15,29),(60,3)):
        values = Tensor.randn(1,32,64).realize()
        pos,count = UOp.variable('start',0,63).bind(start),UOp.variable('rows',1,32).bind(rows)
        block.bounded_mask = False
        expected = original(values,pos,count).pad_to((1,32,64)).numpy()[:,:rows]
        block.bounded_mask = True
        actual = bounded(values,pos,count).pad_to((1,32,64)).numpy()[:,:rows]
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=3e-5)

  def test_symbolic_capacity_preserves_true_rows_and_rebinds(self):
    rng = np.random.default_rng(27)
    source = nn.Linear(64,96,bias=True)
    source.weight.replace(source.weight.half().realize())
    layer = MaliLinear(source)
    run = TinyJit(lambda x,rows:layer(x[:,:rows]).realize())
    weight,bias = source.weight.numpy().astype(np.float32),source.bias.numpy()
    for rows in (32,17,1,31,8,32):
      data = rng.normal(0,.3,(1,32,64)).astype(np.float32)
      result = run(Tensor(data).realize(),UOp.variable('rows',1,32).bind(rows))
      self.assertEqual(result.shape[1].unbind()[1],rows)
      actual = result.pad_to((1,32,96)).numpy()[:,:rows]
      np.testing.assert_allclose(actual,data[:,:rows]@weight.T+bias,atol=2e-6,rtol=2e-5)


@unittest.skipUnless(os.environ.get('MALI_TEST')=='1','explicit OpenCL hardware test')
class TestMaliAttention(unittest.TestCase):
  def test_segmented_embedding_boundaries_and_repeated_configuration(self):
    from types import SimpleNamespace
    from tinygrad import dtypes
    from tinygrad.llm.gemma4 import SegmentedEmbedding
    from tinygrad.llm.kernels.mali import configure, MaliEmbedding
    values = (np.arange(70*12,dtype=np.float32).reshape(70,12)/32).astype(np.float16)
    side = SegmentedEmbedding(70,12,rows=17)
    for index,part in enumerate(side.chunks):
      part.weight.replace(Tensor(values[index*17:index*17+part.weight.shape[0]],device='CL').realize())
    main = nn.Embedding(70,12)
    main.weight.replace(Tensor(values,device='CL').realize())
    model = SimpleNamespace(token_embd=main,per_layer_token_embd=side,config=SimpleNamespace(max_context=32),blk=[])
    first = configure(model,pack_weights=False)
    installed = tuple(side.chunks)
    second = configure(model,pack_weights=False)
    self.assertEqual(first['segmented_embedding_chunks'],5)
    self.assertEqual(second['segmented_embedding_chunks'],5)
    self.assertTrue(all(isinstance(chunk,MaliEmbedding) for chunk in side.chunks))
    self.assertTrue(all(a is b for a,b in zip(installed,side.chunks)))
    from unittest.mock import patch
    from tinygrad.llm.kernels import mali_linear
    with patch.object(mali_linear,'embedding',wraps=mali_linear.embedding) as direct:
      np.testing.assert_array_equal(model.token_embd(Tensor([[0]],device='CL')).numpy()[0,0],values[0].astype(np.float32))
      direct.assert_called_once()
    for dtype in (dtypes.int,dtypes.int64):
      ids = np.array([[0,16,17,33,34,67,68,69,-1,70]])
      expected = np.zeros((1,10,12),np.float32)
      expected[0,:8] = values[ids[0,:8]].astype(np.float32)
      np.testing.assert_array_equal(side(Tensor(ids,device='CL',dtype=dtype)).numpy(),expected)
      with patch('tinygrad.llm.kernels.mali_linear.embedding',side_effect=AssertionError('single-token auxiliary lookup must use ordinary fusion')):
        for token in (0,17,69,-1,70):
          expected_row = values[token].astype(np.float32) if 0<=token<70 else np.zeros(12,np.float32)
          actual = side(Tensor([[token]],device='CL',dtype=dtype)).numpy()
          np.testing.assert_array_equal(actual[0,0],expected_row)

  def test_q8_head_preserves_fp16_gemv_operands_bitwise(self):
    from tinygrad.llm.kernels.mali_linear import gemv, gemv_q8
    rng = np.random.default_rng(84)
    for columns,depth in ((113,128),(2048,1024),(151936,1024)):
      scales = rng.uniform(1e-7,.001,(columns,depth//32)).astype(np.float16)
      coefficients = rng.integers(-128,128,(columns,depth//32,32),dtype=np.int8)
      data = np.empty((columns,depth//32,34),np.uint8)
      data[:,:,:2] = scales.view(np.uint8).reshape(columns,depth//32,2)
      data[:,:,2:] = coefficients.view(np.uint8)
      weight = (scales.astype(np.float32)[:,:,None]*coefficients.astype(np.float32)).astype(np.float16).reshape(columns,depth)
      values = rng.normal(0,.3,(1,1,depth)).astype(np.float32)
      x,raw = Tensor(values,device='CL').realize(),Tensor(data.flatten(),device='CL').realize()
      expected = gemv(x,Tensor(weight,device='CL').realize()).numpy()
      actual = gemv_q8(x,raw,columns,depth).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

  def test_joint_projections_match_logits_and_invalidate_after_weight_change(self):
    from tinygrad.llm.kernels.mali import configure, MaliJoint
    Tensor.manual_seed(19)
    config = TransformerConfig(2,64,128,4,2,1e-5,96,16,10000.,16,16,max_context=40960)
    baseline,fused = Transformer(config),Transformer(config)
    for parameter in nn.state.get_parameters(baseline):parameter.replace(parameter.half().to('CL').realize())
    for parameter in nn.state.get_parameters(fused):parameter.to_('CL')
    nn.state.load_state_dict(fused,nn.state.get_state_dict(baseline),verbose=False)
    configure(baseline,capacities=(32,))
    result = configure(fused,capacities=(32,),fuse_projections=True)
    self.assertEqual((result['fused_qkv'],result['fused_gate_up']),(2,2))
    self.assertIsInstance(fused.blk[0].attn_qkv,MaliJoint)
    installed = fused.blk[0].attn_qkv
    repeated = configure(fused,capacities=(32,),fuse_projections=True)
    self.assertEqual(repeated['fused_qkv'],2)
    self.assertIs(fused.blk[0].attn_qkv,installed)
    old = TinyJit(lambda x,rows,start:baseline.logits_from_embeddings(x[:,:rows],start).realize())
    new = TinyJit(lambda x,rows,start:fused.logits_from_embeddings(x[:,:rows],start).realize())
    rng = np.random.default_rng(39)
    for rows,start in ((17,0),(31,17),(1,32767),(7,40953),(32,40928)):
      values = Tensor(rng.normal(0,.3,(1,32,64)).astype(np.float32),device='CL').realize()
      count = UOp.variable('rows',1,32).bind(rows)
      position = UOp.variable('start',0,40959).bind(start)
      np.testing.assert_array_equal(old(values,count,position).numpy().view(np.uint32),new(values,count,position).numpy().view(np.uint32))
    joint = fused.blk[0].attn_qkv
    self.assertTrue(all(layer.packed_weight is None for layer in joint.layers))
    joint.layers[1].weight.replace(Tensor.zeros_like(joint.layers[1].weight).realize())
    values = Tensor.ones(1,1,64,device='CL').realize()
    np.testing.assert_array_equal(joint(values)[1].numpy(),np.zeros((1,1,32),np.float32))

  def test_qualified_dispatch_matches_legacy_order_at_16k_and_model_max(self):
    from tinygrad.llm.kernels.mali_attention import attention
    from extra.rockchip.profile_mali_attention_order import runner
    rng = np.random.default_rng(125)
    for context in (16384,40960):
      qdata = rng.normal(0,.3,(1,16,64,128)).astype(np.float32)
      data = rng.normal(0,.3,(2,1,8,context,128)).astype(np.float16)
      data[:,:,:,context-1:] = np.nan
      q,kv = Tensor(qdata,device='CL').realize(),Tensor(data,device='CL').realize()
      legacy = runner(16,8,128,context,64,('part','kv','row'))
      current = TinyJit(lambda q,k,end,rows:attention(q[:,:,:rows],k,end,subgroups=True).realize())
      for count in (64,63,7,1):
        end = Tensor([context-1],device='CL',dtype='int32').realize()
        rows = Tensor([count],device='CL',dtype='int32').realize()
        expected = legacy(q,kv,end,rows).numpy()[:,:,:count]
        actual = current(q,kv,UOp.variable('end',1,context).bind(context-1),UOp.variable('rows',1,64).bind(count))
        result = actual.pad_to(qdata.shape).numpy()[:,:,:count]
        np.testing.assert_array_equal(result.view(np.uint32),expected.view(np.uint32))

  def test_two_layers_share_scratch_without_corrupting_dependent_attention(self):
    from tinygrad.llm.kernels.mali import configure
    Tensor.manual_seed(48)
    config = TransformerConfig(2,64,128,4,2,1e-5,96,16,10000.,16,16,max_context=8192)
    reference,gpu = Transformer(config),Transformer(config)
    for parameter in nn.state.get_parameters(reference):parameter.replace(parameter.half().to('CPU').realize())
    for parameter in nn.state.get_parameters(gpu):parameter.to_('CL')
    nn.state.load_state_dict(gpu,nn.state.get_state_dict(reference),verbose=False)
    configure(gpu,capacities=(32,))
    self.assertIs(gpu.blk[0].prefill_attention,gpu.blk[1].prefill_attention)
    expected_run = TinyJit(lambda x,rows:reference.logits_from_embeddings(x[:,:rows],0).realize())
    actual_run = TinyJit(lambda x,rows:gpu.logits_from_embeddings(x[:,:rows],0).realize())
    rng = np.random.default_rng(44)
    for rows in (17,31,1,7,32):
      values = rng.normal(0,.3,(1,32,64)).astype(np.float32)
      count = UOp.variable('rows',1,32).bind(rows)
      expected = expected_run(Tensor(values,device='CPU').realize(),count).numpy()
      actual = actual_run(Tensor(values,device='CL').realize(),count).numpy()
      np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)

  def test_direct_embedding_and_coalesced_gemv(self):
    from tinygrad.llm.kernels.mali_linear import embedding, gemv, gemm
    rng = np.random.default_rng(84)
    for columns,depth in ((113,128),(2048,1024),(151936,1024)):
      weight = (rng.standard_normal((columns,depth),dtype=np.float32)/np.float32(np.sqrt(depth))).astype(np.float16)
      stored = Tensor(weight,device='CL').realize()
      for _ in range(3):
        values = rng.normal(0,.3,(1,1,depth)).astype(np.float32)
        actual = gemv(Tensor(values,device='CL').realize(),stored).numpy()
        expected = values@weight.astype(np.float32).T
        np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=2e-4)
    for rows,columns,depth in ((32,113,128),(64,3072,1024),(7,97,132)):
      weight = (rng.standard_normal((columns,depth),dtype=np.float32)/np.float32(np.sqrt(depth))).astype(np.float16)
      values = rng.normal(0,.3,(1,rows,depth)).astype(np.float32)
      actual = gemm(Tensor(values,device='CL').realize(),Tensor(weight,device='CL').realize()).numpy()
      np.testing.assert_allclose(actual,values@weight.astype(np.float32).T,atol=5e-6,rtol=2e-4)
    weight = rng.normal(0,.3,(300,64)).astype(np.float16)
    stored = Tensor(weight,device='CL').realize()
    for dtype in (np.int32,np.int64):
      ids = np.array([[0,299,2,-1,300,119]],dtype=dtype)
      actual = embedding(Tensor(ids,device='CL').realize(),stored).numpy()
      expected = np.zeros((1,6,64),np.float32)
      expected[0,[0,1,2,5]] = weight[ids[0,[0,1,2,5]]].astype(np.float32)
      np.testing.assert_array_equal(actual,expected)

  def test_causal_gqa_near_capacity_with_rebinding_and_masked_future(self):
    from tinygrad.llm.kernels.mali_attention import attention
    rng = np.random.default_rng(57)
    for context,capacity,dim,heads,kv_heads in ((257,16,16,4,2),(257,16,16,8,2),(257,16,16,16,2),
                                             (8192,32,128,4,2),(40960,8,64,3,1),(40960,1,128,16,8)):
      run = TinyJit(lambda q,kv,end,rows:attention(q[:,:,:rows],kv,end,subgroups=True).realize())
      for index in range(5):
        rows = max(1,capacity-index*3)
        end = context-1-index
        query = rng.normal(0,.3,(1,heads,capacity,dim)).astype(np.float32)
        values = rng.normal(0,.3,(2,1,kv_heads,context,dim)).astype(np.float16)
        values[:,:,:,end:] = np.nan
        if index==1:query *= 100
        expected = np.empty((1,heads,rows,dim),np.float32)
        for head in range(heads):
          for row in range(rows):
            used = end-rows+row+1
            scores = values[0,0,head//(heads//kv_heads),:used].astype(np.float32)@query[0,head,row]/np.float32(np.sqrt(dim))
            probability = np.exp(scores-scores.max())
            probability /= probability.sum()
            expected[0,head,row] = probability@values[1,0,head//(heads//kv_heads),:used].astype(np.float32)
        output = run(Tensor(query,device='CL').realize(),Tensor(values,device='CL').realize(),
                     UOp.variable('end',1,context).bind(end),UOp.variable('rows',1,capacity).bind(rows))
        actual = output.pad_to((1,heads,capacity,dim)).numpy()[:,:,:rows]
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
