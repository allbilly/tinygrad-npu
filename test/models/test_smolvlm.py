import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.model import TransformerConfig
from tinygrad.llm.vision import VisionConfig, VisionEmbeddings, VisionAttention, SmolVLM, pixel_shuffle

class TestSmolVLM(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(11)
    self.vc = VisionConfig(8,16,2,1,8,2,1e-6,2)
    self.tc = TransformerConfig(1,8,16,2,1,1e-5,32,4,10000.,4,4,max_context=32)

  def test_patch_embedding_matches_convolution(self):
    emb = VisionEmbeddings(self.vc)
    image = Tensor.randn(2,3,8,8)
    weight, bias = emb.patch_embedding.weight.numpy(), emb.patch_embedding.bias.numpy()
    expected = np.empty((2,16,8),np.float32)
    for y in range(4):
      for x in range(4):
        patch = image.numpy()[:,:,2*y:2*y+2,2*x:2*x+2].reshape(2,-1)
        expected[:,y*4+x] = patch @ weight.T + bias + emb.position_embedding.weight.numpy()[y*4+x]
    np.testing.assert_allclose(emb(image).numpy(),expected,atol=2e-6,rtol=2e-6)

  def test_pixel_shuffle_spatial_order(self):
    image = np.arange(2*8*8*3,dtype=np.float32).reshape(2,8,8,3)
    # Each output pixel contains rows then columns of its original 2x2 block.
    expected = np.stack([image[:,y:y+2,x:x+2].reshape(2,-1) for y in range(0,8,2) for x in range(0,8,2)],axis=1)
    np.testing.assert_array_equal(pixel_shuffle(Tensor(image.reshape(2,64,3)),2).numpy(),expected)
    for shape,scale in [((1,15,8),2),((1,16,8),3),((1,16,8),0)]:
      with self.assertRaises(ValueError): pixel_shuffle(Tensor.zeros(*shape),scale)

  def test_attention_is_bidirectional(self):
    attn = VisionAttention(self.vc)
    source = Tensor.randn(1,4,8).numpy()
    before = attn(Tensor(source)).numpy()
    source[:,-1] += np.arange(8,dtype=np.float32)
    after = attn(Tensor(source)).numpy()
    self.assertGreater(float(np.max(abs(before[:,0]-after[:,0]))),1e-3)

  def test_image_prompt_substitution(self):
    model = SmolVLM(self.vc,self.tc,31)
    tokens = [1,2,31,31,31,31,3]
    image = Tensor.randn(1,4,8)
    embeds = model.embed_prompt(tokens,image).numpy()
    np.testing.assert_array_equal(embeds[:,2:6],image.numpy())
    np.testing.assert_array_equal(embeds[:,:2],model.text.token_embd(Tensor([[1,2]])).numpy())
    np.testing.assert_array_equal(embeds[:,6:],model.text.token_embd(Tensor([[3]])).numpy())
    for ids in ([1,2],[31]*3,[31,31,1,31,31],[31]*5):
      with self.assertRaises(ValueError): model.embed_prompt(ids,image)
    with self.assertRaises(ValueError): model.encode_image(Tensor.zeros(1,3,8,7))

  def test_decoder_embedding_entry_matches_token_entry(self):
    model = SmolVLM(self.vc,self.tc,31).text
    tokens = Tensor([[1,3,5,7]])
    normal = model.logits(tokens,0).numpy()
    embedded = model.logits_from_embeddings(model.token_embd(tokens).float(),0).numpy()
    np.testing.assert_array_equal(normal,embedded)
    with self.assertRaises(ValueError): model.logits_from_embeddings(Tensor.zeros(1,8),0)

  def test_prompt_embedding_rebinding_and_output_ownership(self):
    model = SmolVLM(self.vc,self.tc,31)
    saved = []
    for dtype in (dtypes.half,dtypes.float):
      model.text.token_embd.weight.replace(model.text.token_embd.weight.cast(dtype).contiguous().realize())
      for iteration in range(7):
        tokens = [iteration,2,31,31,31,31,3]
        image = Tensor.randn(1,4,8).realize()
        output = model.embed_prompt(tokens,image)
        expected = model.text.token_embd.weight.numpy()[tokens].astype(np.float32)[None]
        expected[:,2:6] = image.numpy()
        np.testing.assert_array_equal(output.numpy(),expected)
        saved.append((output,expected.copy()))
    # Arbitrary layouts use the same prepared kernel; earlier outputs stay valid.
    for length in range(8,20):
      model.embed_prompt([1,2,31,31,31,31]+[3]*(length-6),Tensor.randn(1,4,8).realize())
    for output,expected in saved:np.testing.assert_array_equal(output.numpy(),expected)

  def test_greedy_embedding_entry_matches_full_logits_after_rebinding(self):
    from tinygrad.llm.kernels.rockchip_greedy import greedy_argmax
    model = SmolVLM(self.vc,self.tc,31).text
    model.rockchip_greedy = greedy_argmax
    selected = []
    for _ in range(7):
      values = Tensor.randn(1,7,8).realize()
      expected = int(model.logits_from_embeddings(values,0).numpy().argmax())
      actual = int(model.greedy_embeddings_prefill_jit(values).item())
      self.assertEqual(actual,expected)
      selected.append(actual)
    self.assertGreater(len(set(selected)),1)

  def test_prompt_embedding_remains_capturable(self):
    model = SmolVLM(self.vc,self.tc,31)
    tokens = [1,2,31,31,31,31,3]
    run = TinyJit(lambda image:model.embed_prompt(tokens,image).realize())
    for _ in range(6):
      image = Tensor.randn(1,4,8).realize()
      expected = model.text.token_embd.weight.numpy()[tokens].astype(np.float32)[None]
      expected[:,2:6] = image.numpy()
      np.testing.assert_array_equal(run(image).numpy(),expected)

  def test_sampled_embedding_entry_preserves_sampling_and_rng_replay(self):
    model = SmolVLM(self.vc,self.tc,31).text
    values = Tensor.randn(1,7,8).realize()
    for _ in range(4): model.sampled_embeddings_prefill_jit(values,Tensor([.6]).realize()).realize()
    counter = Tensor._device_rng_counters[values.device]
    for temperature in (.05,.6,2.,0.,.6,1.2):
      values = Tensor.randn(1,7,8).realize()
      logits = model.logits_from_embeddings(values,0).realize()
      temp = Tensor([temperature],device=values.device).realize()
      before = counter.numpy().copy()
      expected = (logits/temp.maximum(1e-12) - Tensor.rand_like(logits).maximum(1e-12).log().neg().log()).argmax(-1).item()
      after = counter.numpy().copy()
      counter.assign(Tensor(before,device=values.device)).realize()
      actual = model.sampled_embeddings_prefill_jit(values,temp).item()
      self.assertEqual(actual,expected)
      np.testing.assert_array_equal(counter.numpy(),after)
      self.assertFalse(np.array_equal(before,after))

  def test_staged_encoder_preserves_image_dependence(self):
    from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision
    model = SmolVLM(self.vc,self.tc,31)
    staged = StagedVision(model,attention_device='CPU',native=False)
    previous = None
    for _ in range(4):
      image = Tensor.randn(1,3,8,8).realize()
      actual = staged(image).numpy()
      np.testing.assert_allclose(actual,model.encode_image(image).numpy(),atol=2e-5,rtol=2e-5)
      if previous is not None: self.assertGreater(float(abs(actual-previous).max()),1e-3)
      previous = actual.copy()

if __name__ == '__main__': unittest.main()
