"""ChatGLM2/3 packed projections and adjacent-pair partial rotary layout.

The original decoder rotates adjacent coordinate pairs in the first half of
each head. The generic decoder rotates two contiguous halves of that prefix.
Reorder both query/key projection rows and biases; leave value and tail rows intact.
"""
from tinygrad import Tensor


def split_chatglm_ffn_entries(entries:dict[str,tuple[Tensor,tuple[int,...],int]], hidden_dim:int) -> dict[str,tuple[Tensor,tuple[int,...],int]]:
  """Split file-backed gate/up rows before allocating their final device weights."""
  from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT
  if hidden_dim<1:raise ValueError('Invalid ChatGLM feed-forward dimension')
  result = {}
  for name,(data,shape,typ) in entries.items():
    gate = name.replace('.ffn_up.','.ffn_gate.')
    if not name.endswith('.ffn_up.weight') or gate in entries:
      result[name] = (data,shape,typ)
      continue
    if len(shape)!=2 or shape[0]!=2*hidden_dim:raise ValueError('Invalid ChatGLM packed gate/up shape')
    if typ in _GGML_QUANT:block,block_bytes = _GGML_QUANT[typ]
    elif typ in _GGML_NATIVE:block,block_bytes = 1,_GGML_NATIVE[typ].itemsize
    else:raise ValueError('Unsupported ChatGLM packed gate/up storage')
    if shape[1]<1 or shape[1]%block:raise ValueError('ChatGLM gate/up rows must align to storage blocks')
    cut = hidden_dim*shape[1]//block*block_bytes
    if data.numel()<2*cut:raise ValueError('Invalid ChatGLM packed gate/up storage length')
    half_shape = (hidden_dim,shape[1])
    result[gate] = (data[:cut],half_shape,typ)
    result[name] = (data[cut:2*cut],half_shape,typ)
  return result


def unpack_chatglm(state:dict[str,Tensor], blocks:int, heads:int, kv_heads:int, head_dim:int,
                   rope_dim:int, hidden_dim:int) -> dict[str,Tensor]:
  if min(heads,kv_heads,head_dim,hidden_dim)<1 or heads%kv_heads or not 0<=rope_dim<=head_dim or rope_dim%2:
    raise ValueError('Invalid ChatGLM head dimensions')
  def half_split(value:Tensor,count:int) -> Tensor:
    if not rope_dim:return value
    tail = value.shape[1:]
    per_head = value.reshape(count,head_dim,*tail)
    rotated = per_head[:,:rope_dim].reshape(count,rope_dim//2,2,*tail).transpose(1,2).reshape(count,rope_dim,*tail)
    return rotated.cat(per_head[:,rope_dim:],dim=1).reshape(value.shape)
  for index in range(blocks):
    prefix = f'blk.{index}.'
    for suffix in ('weight','bias'):
      if (key:=prefix+'attn_qkv.'+suffix) not in state:continue
      value = state.pop(key)
      sizes = (heads*head_dim,kv_heads*head_dim,kv_heads*head_dim)
      if value.shape[0]!=sum(sizes):raise ValueError('Invalid ChatGLM packed QKV shape')
      if heads==kv_heads:
        per_head = value.reshape(heads,3,head_dim,*value.shape[1:])
        q,k,v = (per_head[:,i].reshape(heads*head_dim,*value.shape[1:]) for i in range(3))
      else:q,k,v = value.split(sizes,dim=0)
      state[prefix+'attn_q.'+suffix],state[prefix+'attn_k.'+suffix] = half_split(q,heads),half_split(k,kv_heads)
      # Q/K permutation allocates new rows. Compact V as well so its small
      # view does not keep the original full fused QKV allocation alive.
      state[prefix+'attn_v.'+suffix] = v.clone()
    if prefix+'ffn_gate.weight' not in state and (key:=prefix+'ffn_up.weight') in state:
      if state[key].shape[0]!=2*hidden_dim:raise ValueError('Invalid ChatGLM packed gate/up shape')
      state[prefix+'ffn_gate.weight'],state[key] = state[key].split((hidden_dim,hidden_dim),dim=0)
  return state
