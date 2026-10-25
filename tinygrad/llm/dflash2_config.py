"""Data-only DFlash2 checkpoint contract; no Hugging Face configuration code is executed."""
from __future__ import annotations
import hashlib, json, math, pathlib, struct
from dataclasses import dataclass
from typing import Any, Mapping
from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType
from tinygrad.llm.model import TransformerConfig
from tinygrad.nn.state import safe_load

DFLASH2_REVISION = "c9aefeb3bccf819f3575521801d95b3ace87f8e7"
DFLASH2_CONFIG_SHA256 = "f290e9af60a11850b747015ae437da91706405a180d85d95c736ee6ad3c6420d"
DFLASH2_WEIGHTS_SHA256 = "a9f6d56744e3a23ea50eaf20fe2115832ae5364ad607a416a48f96b8800fbb8e"
DFLASH2_WEIGHTS_BYTES = 708494328
DFLASH2_TARGET_SHA256 = "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1"

def _int(value:Any, name:str, minimum:int=1) -> int:
  if type(value) is not int or value < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
  return value

def _float(value:Any, name:str) -> float:
  if type(value) not in (float, int) or not math.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and positive")
  return float(value)

def _object(value:Any, name:str) -> dict[str,Any]:
  if not isinstance(value, dict) or any(not isinstance(k,str) for k in value): raise ValueError(f"{name} must be an object")
  return value

def _unique_object(items:list[tuple[str,Any]]) -> dict[str,Any]:
  result:dict[str,Any] = {}
  for key,value in items:
    if key in result: raise ValueError(f"Duplicate JSON key: {key}")
    result[key] = value
  return result

@dataclass(frozen=True)
class DFlash2Config:
  hidden_size:int
  intermediate_size:int
  num_layers:int
  num_attention_heads:int
  num_key_value_heads:int
  head_dim:int
  norm_eps:float
  vocab_size:int
  rope_theta:float
  layer_types:tuple[str,...]
  sliding_window:int
  sliding_window_non_causal:bool
  aux_hidden_state_layer_ids:tuple[int,...]
  block_size:int
  mask_token_id:int
  conv_kernel_size:int
  conv_group_size:int
  selector_rank:int
  selector_top_k:int
  sample_from_anchor:bool
  target_hidden_size:int
  verifier_name:str
  max_position_embeddings:int

  @property
  def capture_layers(self) -> tuple[int,...]:
    # vLLM e7bb7e2f: qwen3_next.py721,731-733 captures embedding at 0 and block i output at i+1.
    # interfaces.py1647-1649 captures hidden+residual, before the final output norm.
    return tuple(i-1 for i in self.aux_hidden_state_layer_ids)

  @classmethod
  def from_dict(cls, data:Mapping[str,Any]) -> DFlash2Config:
    try:
      if data["architectures"] != ["DFlash2DraftModel"] or data["speculators_model_type"] != "dflash2":
        raise ValueError("Expected a DFlash2DraftModel checkpoint")
      spec = _object(data["speculators_config"], "speculators_config")
      if spec["algorithm"] != "dflash2" or spec["default_proposal_method"] != "greedy":
        raise ValueError("Only trained DFlash2 greedy proposals are supported")
      verifier = _object(spec["verifier"], "verifier")
      if verifier["architectures"] != ["Qwen3_5ForConditionalGeneration"] or not isinstance(verifier["name_or_path"],str):
        raise ValueError("Unsupported DFlash2 verifier")
      layer = _object(data["transformer_layer_config"], "transformer_layer_config")
      if layer["model_type"] != "qwen3" or layer["hidden_act"] != "silu": raise ValueError("Unsupported draft layer architecture")
      if layer["attention_bias"] is not False or layer["attention_dropout"] != 0: raise ValueError("Unsupported attention bias or dropout")
      if data["sample_from_anchor"] is not False: raise ValueError("Only the trained bonus-anchor alignment is supported")
      if data["tie_word_embeddings"] is not False: raise ValueError("Unexpected tied draft embeddings")
      if type(data["sliding_window_non_causal"]) is not bool: raise ValueError("sliding_window_non_causal must be boolean")
      rope = _object(layer["rope_parameters"], "rope_parameters")
      if rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 1: raise ValueError("Unsupported draft RoPE")
      dim = _int(layer["hidden_size"], "hidden_size")
      heads = _int(layer["num_attention_heads"], "num_attention_heads")
      kv_heads = _int(layer["num_key_value_heads"], "num_key_value_heads")
      head_dim = _int(layer["head_dim"], "head_dim")
      num_layers = _int(layer["num_hidden_layers"], "num_hidden_layers")
      vocab = _int(layer["vocab_size"], "vocab_size")
      block = _int(data["block_size"], "block_size", 2)
      group = _int(data["conv_group_size"], "conv_group_size")
      kernel = _int(data["conv_kernel_size"], "conv_kernel_size")
      top_k = _int(data["selector_top_k"], "selector_top_k")
      mask = _int(data["mask_token_id"], "mask_token_id", 0)
      taps = data["aux_hidden_state_layer_ids"]
      if not isinstance(taps,list) or not taps or any(type(i) is not int or i < 1 for i in taps):
        raise ValueError("Auxiliary IDs must name post-block states (capture IDs >= 1)")
      if taps != sorted(set(taps)): raise ValueError("Auxiliary capture IDs must be unique and increasing")
      types = layer["layer_types"]
      if not isinstance(types,list) or len(types) != num_layers or any(x not in ("sliding_attention","full_attention") for x in types):
        raise ValueError("Invalid draft attention layer types")
      target_dim = dim if data.get("target_hidden_size") is None else _int(data["target_hidden_size"], "target_hidden_size")
      if target_dim != dim: raise ValueError("This DFlash2 architecture requires matching target and draft hidden sizes")
      if heads % kv_heads or head_dim % 2 or dim % group or kernel > block or top_k > vocab or mask >= vocab:
        raise ValueError("Inconsistent DFlash2 attention, convolution, selector, or vocabulary dimensions")
      if _int(data["draft_vocab_size"], "draft_vocab_size") != vocab: raise ValueError("Reduced draft vocabulary is unsupported")
      proposals = spec["proposal_methods"]
      if proposals != [{"accept_tolerance":0.0,"proposal_type":"greedy","speculative_tokens":block-1,"verifier_accept_k":1}]:
        raise ValueError("Unsupported proposal or verifier acceptance configuration")
      return cls(dim, _int(layer["intermediate_size"],"intermediate_size"), num_layers, heads, kv_heads, head_dim,
                 _float(layer["rms_norm_eps"],"rms_norm_eps"), vocab, _float(rope["rope_theta"],"rope_theta"), tuple(types),
                 _int(layer["sliding_window"],"sliding_window"), data["sliding_window_non_causal"], tuple(taps), block, mask, kernel,
                 group, _int(data["selector_rank"],"selector_rank"), top_k, False, target_dim, verifier["name_or_path"],
                 _int(layer["max_position_embeddings"],"max_position_embeddings"))
    except KeyError as error: raise ValueError(f"Missing DFlash2 configuration field: {error.args[0]}") from error

  def validate_target(self, target:TransformerConfig) -> None:
    if target.dim != self.target_hidden_size or target.vocab_size != self.vocab_size:
      raise ValueError("DFlash2 target hidden size or vocabulary does not match")
    if max(self.aux_hidden_state_layer_ids) > target.num_blocks: raise ValueError("DFlash2 auxiliary capture exceeds target depth")
    if target.hc is not None or target.rwkv7: raise ValueError("Unsupported target architecture")

  def weight_shapes(self) -> dict[str,tuple[int,...]]:
    dim,rank,vocab = self.hidden_size,self.selector_rank,self.vocab_size
    result = {"fc.weight":(dim,len(self.capture_layers)*self.target_hidden_size), "hidden_norm.weight":(dim,), "norm.weight":(dim,),
              "candidate_selector.hidden_projection.weight":(rank,dim), "candidate_selector.predecessor_codebook":(vocab,rank),
              "candidate_selector.successor_codebook":(vocab,rank)}
    for i in range(self.num_layers):
      prefix = f"layers.{i}."
      shapes = {"input_layernorm.weight":(dim,), "post_attention_layernorm.weight":(dim,),
                "mlp.gate_proj.weight":(self.intermediate_size,dim), "mlp.up_proj.weight":(self.intermediate_size,dim),
                "mlp.down_proj.weight":(dim,self.intermediate_size),
                "self_attn.q_proj.weight":(self.num_attention_heads*self.head_dim,dim),
                "self_attn.k_proj.weight":(self.num_key_value_heads*self.head_dim,dim),
                "self_attn.v_proj.weight":(self.num_key_value_heads*self.head_dim,dim),
                "self_attn.o_proj.weight":(dim,self.num_attention_heads*self.head_dim),
                "self_attn.q_norm.weight":(self.head_dim,), "self_attn.k_norm.weight":(self.head_dim,)}
      for name in ("attention_conv","mlp_conv"):
        shapes[f"{name}.base_kernel"] = (2,self.conv_kernel_size,dim)
        shapes[f"{name}.kernel_projection.weight"] = (2*self.conv_kernel_size*(dim//self.conv_group_size),dim)
      result.update({prefix+key:value for key,value in shapes.items()})
    return result

  def validate_weights(self, weights:Mapping[str,Tensor]) -> None:
    expected = self.weight_shapes()
    if weights.keys() != expected.keys():
      missing,extra = sorted(expected.keys()-weights.keys()),sorted(weights.keys()-expected.keys())
      raise ValueError(f"DFlash2 tensor names differ: missing={missing}, extra={extra}")
    for name,shape in expected.items():
      if weights[name].shape != shape: raise ValueError(f"DFlash2 tensor {name} has shape {weights[name].shape}, expected {shape}")
      if weights[name].dtype not in (dtypes.bfloat16,dtypes.float16,dtypes.float32): raise ValueError(f"Non-floating DFlash2 tensor: {name}")

def _sha256(path:pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8*1024*1024): digest.update(chunk)
  return digest.hexdigest()

def _validate_header(path:pathlib.Path, config:DFlash2Config) -> None:
  with path.open("rb") as stream:
    raw = stream.read(8)
    if len(raw) != 8: raise ValueError("Truncated safetensors size")
    size, = struct.unpack("<Q",raw)
    if not 2 <= size <= 16*1024*1024 or size+8 > path.stat().st_size: raise ValueError("Invalid safetensors header size")
    header = _object(json.loads(stream.read(size),object_pairs_hook=_unique_object),"safetensors header")
  tensors = {name:item for name,item in header.items() if name != "__metadata__"}
  if tensors.keys() != config.weight_shapes().keys(): raise ValueError("Safetensors tensor names do not match the trained DFlash2 architecture")
  ranges = []
  for name,shape in config.weight_shapes().items():
    item = _object(tensors[name],name)
    if set(item) != {"dtype","shape","data_offsets"} or item["dtype"] != "BF16" or item["shape"] != list(shape):
      raise ValueError(f"Invalid trained tensor metadata: {name}")
    offsets = item["data_offsets"]
    if not isinstance(offsets,list) or len(offsets)!=2 or any(type(i) is not int or i<0 for i in offsets):
      raise ValueError(f"Invalid trained tensor offsets: {name}")
    start,end = offsets
    if end-start != math.prod(shape)*2: raise ValueError(f"Invalid trained tensor byte count: {name}")
    ranges.append((start,end))
  current = 0
  for start,end in sorted(ranges):
    if start != current: raise ValueError("Safetensors has overlapping or noncontiguous tensor data")
    current = end
  if size+8+current != path.stat().st_size: raise ValueError("Safetensors data is truncated or has trailing bytes")

def load_dflash2(directory:str|pathlib.Path, target:TransformerConfig, *, target_checkpoint:str|pathlib.Path,
                 device:str|None=None, dtype:DType|None=None) -> tuple[DFlash2Config,dict[str,Tensor]]:
  """Load only the pinned, verified 2B draft/target pair. Generic dimensions do not prove token-ID compatibility.

  Hashes cover the actual local files before parsing/loading tensors. BF16 DISK tensors remain lazy unless a device is requested.
  Passing dtype changes arithmetic precision and requires separate output/speed qualification.
  """
  root = pathlib.Path(directory)
  config_path,weight_path = root/"config.json",root/"model.safetensors"
  if config_path.stat().st_size > 1024*1024 or _sha256(config_path) != DFLASH2_CONFIG_SHA256:
    raise ValueError("DFlash2 configuration does not match the pinned checkpoint")
  config = DFlash2Config.from_dict(_object(json.loads(config_path.read_bytes(),object_pairs_hook=_unique_object),"config"))
  config.validate_target(target)
  if target.num_blocks != 24 or config.verifier_name != "Qwen/Qwen3.5-2B": raise ValueError("Unexpected pinned verifier architecture")
  if _sha256(pathlib.Path(target_checkpoint)) != DFLASH2_TARGET_SHA256:
    raise ValueError("Target checkpoint/tokenizer identity does not match the qualified DFlash2 pair")
  if weight_path.stat().st_size != DFLASH2_WEIGHTS_BYTES or _sha256(weight_path) != DFLASH2_WEIGHTS_SHA256:
    raise ValueError("DFlash2 weights do not match the pinned checkpoint")
  _validate_header(weight_path,config)
  if dtype is not None and dtype not in (dtypes.bfloat16,dtypes.float16,dtypes.float32): raise ValueError("Unsupported draft compute dtype")
  weights = safe_load(weight_path)
  config.validate_weights(weights)
  if device is not None: weights = {name:value.to(device).cast(dtype or value.dtype).realize() for name,value in weights.items()}
  elif dtype is not None: raise ValueError("A compute device is required when converting draft dtype")
  return config,weights
