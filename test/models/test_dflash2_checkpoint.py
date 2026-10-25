import copy, json, pathlib, struct, tempfile, unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.dflash2_config import DFlash2Config, _validate_header, load_dflash2
from tinygrad.llm.dflash2_selector import CandidateSelector
from tinygrad.llm.model import TransformerConfig

def fixture():
  # Independent small version of the published data-only config, including its trained anchor/proposal contract.
  return {"architectures":["DFlash2DraftModel"],"speculators_model_type":"dflash2","aux_hidden_state_layer_ids":[1,3],
          "block_size":4,"conv_group_size":2,"conv_kernel_size":2,"draft_vocab_size":11,"mask_token_id":10,
          "sample_from_anchor":False,"selector_rank":3,"selector_top_k":3,"sliding_window_non_causal":True,
          "target_hidden_size":None,"tie_word_embeddings":False,
          "speculators_config":{"algorithm":"dflash2","default_proposal_method":"greedy",
            "proposal_methods":[{"accept_tolerance":0.0,"proposal_type":"greedy","speculative_tokens":3,"verifier_accept_k":1}],
            "verifier":{"architectures":["Qwen3_5ForConditionalGeneration"],"name_or_path":"Qwen/Qwen3.5-2B"}},
          "transformer_layer_config":{"model_type":"qwen3","hidden_act":"silu","attention_bias":False,"attention_dropout":0.0,
            "hidden_size":8,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"num_hidden_layers":2,
            "vocab_size":11,"layer_types":["sliding_attention","full_attention"],"rms_norm_eps":1e-6,"sliding_window":16,
            "max_position_embeddings":64,"rope_parameters":{"partial_rotary_factor":1.0,"rope_theta":1e7,"rope_type":"default"}}}

def selector_fixture():
  config = DFlash2Config.from_dict(fixture())
  rng = np.random.default_rng(917)
  arrays = {name:rng.normal(0,.7,shape).astype(np.float32) for name,shape in config.weight_shapes().items()
            if name.startswith("candidate_selector.")}
  return config,arrays,CandidateSelector(config,{name:Tensor(value,device="CPU") for name,value in arrays.items()})

class TestDFlash2Checkpoint(unittest.TestCase):
  def test_capture_mapping_and_exact_tensor_layout(self):
    config = DFlash2Config.from_dict(fixture())
    self.assertEqual(config.capture_layers,(0,2))
    self.assertEqual(len(config.weight_shapes()),36)
    self.assertEqual(config.weight_shapes()["fc.weight"],(8,16))
    self.assertEqual(config.weight_shapes()["layers.0.attention_conv.base_kernel"],(2,2,8))
    weights = {name:Tensor.zeros(shape,device="CPU") for name,shape in config.weight_shapes().items()}
    config.validate_weights(weights)
    for modified in ({**weights,"extra.weight":Tensor.zeros(1)}, {k:v for k,v in weights.items() if k!="norm.weight"},
                     {**weights,"norm.weight":Tensor.zeros(9)}, {**weights,"norm.weight":Tensor.zeros(8,dtype=dtypes.int32)}):
      with self.assertRaises(ValueError): config.validate_weights(modified)
    target = TransformerConfig(3,8,16,2,1,1e-6,11,4,1e7,4,4,max_context=64)
    config.validate_target(target)
    for other in (replace(target,num_blocks=2),replace(target,vocab_size=12),replace(target,dim=16)):
      with self.assertRaises(ValueError): config.validate_target(other)

  def test_reject_unsupported_or_inconsistent_config(self):
    for key,value in (("architectures",["DFlashDraftModel"]),("speculators_model_type","dflash"),("sample_from_anchor",True),
                      ("aux_hidden_state_layer_ids",[0,2]),("aux_hidden_state_layer_ids",[3,1]),("aux_hidden_state_layer_ids",[1,1]),
                      ("aux_hidden_state_layer_ids",[True]),("conv_group_size",3),("conv_kernel_size",5),("selector_top_k",12),
                      ("draft_vocab_size",10),("mask_token_id",11),("block_size",True),("sliding_window_non_causal",1)):
      data = copy.deepcopy(fixture())
      data[key] = value
      with self.subTest(key=key,value=value),self.assertRaises(ValueError): DFlash2Config.from_dict(data)
    data = fixture()
    del data["selector_rank"]
    with self.assertRaises(ValueError): DFlash2Config.from_dict(data)

  def test_header_offsets_and_pinned_identity_before_loading(self):
    config = DFlash2Config.from_dict(fixture())
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory)/"model.safetensors"
      def write(break_offset=False):
        header,offset = {},0
        for i,(name,shape) in enumerate(config.weight_shapes().items()):
          size = int(np.prod(shape))*2
          header[name] = {"dtype":"BF16","shape":list(shape),"data_offsets":[offset+(1 if break_offset and i==1 else 0),offset+size]}
          offset += size
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q",len(raw))+raw+b"\0"*offset)
      write()
      _validate_header(path,config)
      write(True)
      with self.assertRaises(ValueError): _validate_header(path,config)
      (path.parent/"config.json").write_text(json.dumps(fixture()))
      target = TransformerConfig(3,8,16,2,1,1e-6,11,4,1e7,4,4,max_context=64)
      with patch("tinygrad.llm.dflash2_config.safe_load") as safe_load:
        with self.assertRaisesRegex(ValueError,"pinned checkpoint"): load_dflash2(path.parent,target,target_checkpoint=path)
        safe_load.assert_not_called()

class TestDFlash2Selector(unittest.TestCase):
  def test_sparse_scores_and_topk_match_independent_numpy(self):
    config,arrays,selector = selector_fixture()
    rng = np.random.default_rng(49)
    hidden = rng.normal(size=(2,3,config.hidden_size)).astype(np.float32)
    logits = rng.normal(size=(2,3,config.vocab_size)).astype(np.float32)
    predecessor = np.array([[3,4,5],[7,2,1]],dtype=np.int32)
    candidates = np.array([[[2,6,9],[8,0,1],[4,6,0]],[[2,8,1],[0,1,9],[3,5,7]]],dtype=np.int32)
    projected = hidden@arrays["candidate_selector.hidden_projection.weight"].T
    context = arrays["candidate_selector.predecessor_codebook"][predecessor]*projected
    expected = np.take_along_axis(logits,candidates,-1)+(context[...,None,:]*arrays["candidate_selector.successor_codebook"][candidates]).sum(-1)
    actual = selector.score_candidates(Tensor(logits),Tensor(hidden),Tensor(predecessor),Tensor(candidates)).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
    ids,scores = selector.select(Tensor(logits),Tensor(hidden),Tensor(predecessor))
    expected_ids = np.argsort(-logits,axis=-1)[...,:config.selector_top_k]
    np.testing.assert_array_equal(ids.numpy(),expected_ids)
    self.assertEqual(scores.shape,(2,3,3))

  def test_sequential_greedy_walk_skips_untrained_anchor_slot(self):
    config,arrays,selector = selector_fixture()
    rng = np.random.default_rng(187)
    hidden = rng.normal(size=(1,4,8)).astype(np.float32)
    logits = rng.normal(size=(1,4,11)).astype(np.float32)
    logits[0,0,10] = 10000  # Never sampled: slot 0 is the real anchor, not a trained token prediction.
    expected = []
    predecessor = 5
    for slot in (1,2,3):
      ids = np.argsort(-logits[0,slot])[:config.selector_top_k]
      projected = hidden[0,slot]@arrays["candidate_selector.hidden_projection.weight"].T
      context = arrays["candidate_selector.predecessor_codebook"][predecessor]*projected
      scores = logits[0,slot,ids]+arrays["candidate_selector.successor_codebook"][ids]@context
      predecessor = int(ids[np.argmax(scores)])
      expected.append(predecessor)
    for count in range(4):
      self.assertEqual(selector.choose(Tensor(logits),Tensor(hidden),5,count),expected[:count])
    with self.assertRaises(ValueError): selector.choose(Tensor(logits[:,:3]),Tensor(hidden[:,:3]),5,2)

  def test_boundary_ties_have_deterministic_token_order_and_nonfinite_is_rejected(self):
    _,_,selector = selector_fixture()
    logits = Tensor([[3.,2.,2.,2.,1.,0.,-1.,-2.,-3.,-4.,-5.]],device="CPU")
    np.testing.assert_array_equal(selector._topk(logits,3).numpy(),[[0,1,2]])
    with self.assertRaises(ValueError): selector._topk(Tensor([[float("nan")]*11],device="CPU"),3)
    with self.assertRaises(ValueError): selector._topk(logits,12)

  def test_greedy_walk_rejects_malformed_candidate_host_structure(self):
    _,_,selector = selector_fixture()
    logits,hidden = Tensor.zeros(1,4,11,device="CPU"),Tensor.zeros(1,4,8,device="CPU")
    malformed = (1, [], [1], [[]], [[[0,1]]], [[[0,1,True]]], [[[0,1,11]]], [[[0,1,2.0]]])
    for host in malformed:
      with self.subTest(host=host),patch.object(selector,"_topk",return_value=Tensor.zeros(1,1,3,device="CPU",dtype=dtypes.int32)), \
           patch.object(Tensor,"tolist",return_value=host),self.assertRaises(ValueError):
        selector.choose(logits,hidden,5,1)

if __name__ == "__main__": unittest.main()
