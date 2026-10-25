"""Regression tests for preserving attention output gates through custom hooks.

The tests use CPU TransformerBlock objects and deterministic hook outputs. They
intentionally do not configure prepared-QKV or any serving/device placement.
"""
import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.llm.model import TransformerBlock, TransformerConfig


def _block(gated:bool) -> TransformerBlock:
  config=TransformerConfig(num_blocks=1,dim=8,hidden_dim=16,n_heads=2,n_kv_heads=1,norm_eps=1e-5,
    vocab_size=32,head_dim=4,rope_theta=10000.0,rope_dim=4,v_head_dim=4,max_context=8,
    attn_output_gate=gated)
  block=TransformerBlock(config)
  q_weight=np.zeros((16 if gated else 8,8),dtype=np.float32)
  if gated:
    for head in range(2):
      for lane in range(4): q_weight[(head*2+1)*4+lane,head*4+lane]=0.7
  else:
    q_weight[:]=np.eye(8,dtype=np.float32)*0.7
  block.attn_q.weight=Tensor(q_weight,device='CPU').realize()
  block.attn_k.weight=Tensor.zeros(*block.attn_k.weight.shape,device='CPU').realize()
  block.attn_v.weight=Tensor.zeros(*block.attn_v.weight.shape,device='CPU').realize()
  block.attn_output.weight=Tensor(np.eye(8,dtype=np.float32),device='CPU').realize()
  return block


def _context(rows:int) -> np.ndarray:
  values=np.empty((1,2,rows,4),dtype=np.float32)
  for head in range(2):
    for row in range(rows):
      for dim in range(4): values[0,head,row,dim]=0.2*(head+1)+0.07*(row+1)+0.03*(dim+1)
  return values


def _install_hooks(block:TransformerBlock,rows:int) -> list:
  calls=[]
  def hook(kind):
    def attend(q,assigned_kv,end):
      calls.append((kind,end,tuple(q.shape),tuple(assigned_kv.shape)))
      return Tensor(_context(rows),device='CPU').realize()
    return attend
  block.prefill_attention=hook('prefill')
  block.decode_attention=hook('decode')
  return calls


@pytest.mark.parametrize(('rows','phase'),[(3,'prefill'),(1,'decode')])
def test_output_gate_is_applied_after_custom_attention_hook(rows,phase):
  block=_block(gated=True)
  x_values=(np.arange(1,rows*8+1,dtype=np.float32).reshape(1,rows,8)/10).astype(np.float32)
  x=Tensor(x_values,device='CPU').realize()
  block._init_state(x)
  calls=_install_hooks(block,rows)
  actual=block._attention(x,0).numpy()
  assert calls and calls[-1][0]==phase and calls[-1][1]==rows

  # The q projection interleaves [query, gate] within each head. The hook's
  # output is [B,H,T,Dh]; flatten to model width only after applying sigmoid(gate).
  q_projection=block.attn_q(x)
  query_gate=q_projection.reshape(1,rows,2,2,4)[:,:,:,1,:].reshape(1,rows,8)
  hook_context=Tensor(_context(rows),device='CPU').transpose(1,2).reshape(1,rows,8)
  expected=block.attn_output(hook_context*query_gate.sigmoid()).numpy()
  ungated=block.attn_output(hook_context).numpy()
  assert not np.all(query_gate.sigmoid().numpy()==1.0)
  assert not np.array_equal(expected,ungated)
  np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize(('rows','phase'),[(3,'prefill'),(1,'decode')])
def test_ungated_projection_still_uses_custom_attention_hook(rows,phase):
  block=_block(gated=False)
  x_values=(np.arange(1,rows*8+1,dtype=np.float32).reshape(1,rows,8)/10).astype(np.float32)
  x=Tensor(x_values,device='CPU').realize()
  block._init_state(x)
  calls=_install_hooks(block,rows)
  actual=block._attention(x,0).numpy()
  assert calls and calls[-1][0]==phase and calls[-1][1]==rows
  hook_context=Tensor(_context(rows),device='CPU').transpose(1,2).reshape(1,rows,8)
  expected=block.attn_output(hook_context).numpy()
  np.testing.assert_array_equal(actual,expected)
