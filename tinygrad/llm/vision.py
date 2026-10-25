"""SmolVLM-v1 image encoder, connector, and image-conditioned Llama decoder.

The initial image contract is one unpadded square image at the configured native
resolution. Crops, videos, padded batches, and alternate position grids require
their own preprocessing contract; they are not silently treated as square images.
"""
from __future__ import annotations
import json, pathlib, re
from dataclasses import dataclass
from typing import Callable
from tinygrad import Tensor, nn, TinyJit, dtypes, Device
from tinygrad.llm.model import Transformer, TransformerConfig
from tinygrad.llm.kernels.vision_embed import PromptEmbedding
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load

@dataclass(frozen=True)
class VisionConfig:
  dim:int
  hidden_dim:int
  n_heads:int
  num_blocks:int
  image_size:int
  patch_size:int
  norm_eps:float
  scale_factor:int

class VisionAttention:
  def __init__(self, config:VisionConfig):
    self.n_heads, self.head_dim = config.n_heads, config.dim // config.n_heads
    if config.dim % config.n_heads: raise ValueError('Vision dimension must be divisible by head count')
    self.q_proj, self.k_proj, self.v_proj, self.out_proj = [nn.Linear(config.dim, config.dim) for _ in range(4)]
    self.attention:Callable[[Tensor,Tensor,Tensor],Tensor]|None = None

  def __call__(self, x:Tensor) -> Tensor:
    b, t, d = x.shape
    q, k, v = (proj(x).reshape(b,t,self.n_heads,self.head_dim).transpose(1,2) for proj in (self.q_proj,self.k_proj,self.v_proj))
    # Vision attention is bidirectional. No text causal mask or KV cache applies.
    result = self.attention(q,k,v) if self.attention is not None else q.scaled_dot_product_attention(k,v)
    return self.out_proj(result.transpose(1,2).reshape(b,t,d))

class VisionMLP:
  def __init__(self, config:VisionConfig):
    self.fc1, self.fc2 = nn.Linear(config.dim, config.hidden_dim), nn.Linear(config.hidden_dim, config.dim)

  def __call__(self, x:Tensor) -> Tensor: return self.fc2(self.fc1(x).gelu(approximate='tanh'))

class VisionBlock:
  def __init__(self, config:VisionConfig):
    self.self_attn, self.mlp = VisionAttention(config), VisionMLP(config)
    self.layer_norm1, self.layer_norm2 = (nn.LayerNorm(config.dim, config.norm_eps) for _ in range(2))

  def __call__(self, x:Tensor) -> Tensor:
    x = x + self.self_attn(self.layer_norm1(x))
    return x + self.mlp(self.layer_norm2(x))

class VisionEmbeddings:
  def __init__(self, config:VisionConfig):
    self.config = config
    self.patch_embedding = nn.Linear(3 * config.patch_size**2, config.dim)
    self.position_embedding = nn.Embedding((config.image_size // config.patch_size)**2, config.dim)

  def __call__(self, pixels:Tensor) -> Tensor:
    c = self.config
    if pixels.ndim != 4 or pixels.shape[1:] != (3,c.image_size,c.image_size):
      raise ValueError(f'Expected unpadded RGB pixels (batch, 3, {c.image_size}, {c.image_size})')
    b, g, p = pixels.shape[0], c.image_size // c.patch_size, c.patch_size
    patches = pixels.reshape(b,3,g,p,g,p).permute(0,2,4,1,3,5).reshape(b,g*g,3*p*p).contiguous()
    positions = Tensor.arange(g*g).to(pixels.device)
    return self.patch_embedding(patches) + self.position_embedding(positions)

class SigLIPVision:
  def __init__(self, config:VisionConfig):
    self.embeddings = VisionEmbeddings(config)
    self.layers = [VisionBlock(config) for _ in range(config.num_blocks)]
    self.post_layernorm = nn.LayerNorm(config.dim, config.norm_eps)

  def __call__(self, pixels:Tensor) -> Tensor:
    x = self.embeddings(pixels).float()
    for layer in self.layers: x = layer(x).realize()
    return self.post_layernorm(x)

def pixel_shuffle(x:Tensor, scale:int) -> Tensor:
  """Match Idefics3's spatial/channel order, including its intermediate transpose."""
  b, seq, dim = x.shape
  side = int(seq**.5)
  if scale < 1 or side*side != seq or side % scale: raise ValueError('Pixel shuffle requires a divisible square patch grid')
  return x.reshape(b,side,side//scale,dim*scale).permute(0,2,1,3).reshape(
    b,side//scale,side//scale,dim*scale*scale).permute(0,2,1,3).reshape(b,seq//(scale*scale),dim*scale*scale)

class SmolVLM:
  def __init__(self, vision_config:VisionConfig, text_config:TransformerConfig, image_token_id:int):
    self.vision_config, self.image_token_id = vision_config, image_token_id
    self.vision = SigLIPVision(vision_config)
    self.connector = nn.Linear(vision_config.dim * vision_config.scale_factor**2, text_config.dim, bias=False)
    self.text = Transformer(text_config)
    self._prompt_embedding = PromptEmbedding()
    self.encode_jit = TinyJit(lambda pixels: self.encode_image(pixels).realize())

  @property
  def image_tokens(self) -> int: return (self.vision_config.image_size // self.vision_config.patch_size // self.vision_config.scale_factor)**2

  def encode_image(self, pixels:Tensor) -> Tensor:
    return self.connector(pixel_shuffle(self.vision(pixels),self.vision_config.scale_factor)).float()

  def embed_prompt(self, tokens:list[int], image_embeddings:Tensor) -> Tensor:
    """Substitute exactly one contiguous native-image token sequence."""
    if image_embeddings.shape != (1,self.image_tokens,self.text.config.dim): raise ValueError('Incorrect image embedding shape')
    indices = [i for i,t in enumerate(tokens) if t == self.image_token_id]
    if len(indices) != self.image_tokens or indices != list(range(indices[0],indices[0]+self.image_tokens)):
      raise ValueError('Prompt must contain exactly one contiguous image token sequence')
    if len(tokens) > self.text.max_context: raise ValueError('Image prompt exceeds context capacity')
    device, start = self.text.token_embd.weight.device, indices[0]
    if device=='CPU' and self.text.config.dim%4==0:
      from tinygrad.llm.kernels.vision_embed import embed_prompt
      return embed_prompt(tokens,self.text.token_embd.weight,image_embeddings.to('CPU').float(),start,self.image_tokens,self._prompt_embedding)
    parts = []
    if start: parts.append(self.text.token_embd(Tensor([tokens[:start]],device=device)).float())
    parts.append(image_embeddings.to(device).float())
    if start+self.image_tokens < len(tokens):
      parts.append(self.text.token_embd(Tensor([tokens[start+self.image_tokens:]],device=device)).float())
    return Tensor.cat(*parts,dim=1).contiguous()

  @staticmethod
  def from_hf(directory:str|pathlib.Path, max_context:int=1024, dtype=dtypes.half, vision_device:str|None=None) -> SmolVLM:
    root = pathlib.Path(directory)
    cfg = json.loads((root/'config.json').read_text())
    if cfg.get('model_type') != 'idefics3': raise ValueError('Only SmolVLM-v1 (Idefics3) checkpoints are supported')
    v, t = cfg['vision_config'], cfg['text_config']
    if v['hidden_act'] != 'gelu_pytorch_tanh' or t.get('rope_scaling') is not None: raise ValueError('Unsupported SmolVLM configuration')
    if max_context < 1 or max_context > t['max_position_embeddings']: raise ValueError('Invalid context capacity')
    vc = VisionConfig(v['hidden_size'],v['intermediate_size'],v['num_attention_heads'],v['num_hidden_layers'],
                      v['image_size'],v['patch_size'],v['layer_norm_eps'],cfg['scale_factor'])
    tc = TransformerConfig(t['num_hidden_layers'],t['hidden_size'],t['intermediate_size'],t['num_attention_heads'],
                           t['num_key_value_heads'],t['rms_norm_eps'],t['vocab_size'],t['head_dim'],t['rope_theta'],
                           t['head_dim'],t['head_dim'],max_context=max_context)
    model = SmolVLM(vc,tc,cfg['image_token_id'])
    if vision_device is not None:
      for module in (model.vision,model.connector):
        for weight in get_state_dict(module).values(): weight.replace(weight.to(vision_device))
    replacements = {'embed_tokens':'token_embd','norm':'output_norm','input_layernorm':'attn_norm','post_attention_layernorm':'ffn_norm',
                    'self_attn.q_proj':'attn_q','self_attn.k_proj':'attn_k','self_attn.v_proj':'attn_v','self_attn.o_proj':'attn_output',
                    'mlp.gate_proj':'ffn_gate','mlp.up_proj':'ffn_up','mlp.down_proj':'ffn_down'}
    weights = {}
    for key,value in safe_load(root/'model.safetensors').items():
      if value.dtype is dtypes.bfloat16:
        # RK3588's host compiler and Mali OpenCL do not expose a BF16 scalar
        # type. Decode the checkpoint's bits without compiling BF16 operations.
        raw = value.bitcast(dtypes.uint16).to(Device.DEFAULT).realize()
        value = (raw.cast(dtypes.uint32) << 16).bitcast(dtypes.float)
      if key.startswith('model.vision_model.'):
        name = key.replace('model.vision_model.','vision.').replace('encoder.layers.','layers.')
        if name == 'vision.embeddings.patch_embedding.weight': value = value.flatten(1)
      elif key == 'model.connector.modality_projection.proj.weight': name = 'connector.weight'
      elif key == 'lm_head.weight': name = 'text.output.weight'
      elif key.startswith('model.text_model.'):
        name = re.sub(r'^model.text_model.layers\.(\d+)\.',r'text.blk.\1.',key)
        if name == key: name = key.replace('model.text_model.','text.')
        for old,new in replacements.items(): name = name.replace('.'+old+'.','.'+new+'.')
      else: raise ValueError(f'Unexpected SmolVLM weight: {key}')
      weights[name] = value.cast(dtype)
    load_state_dict(model,weights,verbose=False,consume=True)
    return model
