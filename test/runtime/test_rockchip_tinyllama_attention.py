import ctypes, functools, subprocess, tempfile, unittest
from pathlib import Path
import numpy as np
from tinygrad.llm.kernels.rockchip_cpu import _fast_softmax_source

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'tinygrad/llm/kernels/rockchip_causal_attention.c'
HEADS,KV_HEADS,DIM,CONTEXT=32,4,64,512
GROUP=HEADS//KV_HEADS
# The prepared attention order is (query-within-KV-group, KV head).
Q_TO_GROUPED=np.asarray([(h%GROUP)*KV_HEADS+h//GROUP for h in range(HEADS)],dtype=np.intp)

@functools.lru_cache(None)
def compile_helper(row_major:bool,token_major:bool):
  directory=tempfile.TemporaryDirectory(prefix='rockchip-tinyllama-attention-')
  source=Path(directory.name)/'helper.c'
  library=Path(directory.name)/'helper.so'
  macros='\n'.join((
    '#define ATTENTION_HELPER_API','#define HEADS 32','#define KV_HEADS 4','#define DIM 64','#define CONTEXT 512',
    '#define SHARE_KV 1',f'#define ROW_MAJOR {int(row_major)}',f'#define TOKEN_MAJOR_OUTPUT {int(token_major)}'))
  source.write_text(_fast_softmax_source()+'\n'+macros+'\n'+HELPER.read_text())
  subprocess.run(['clang','-O2','-shared','-fPIC','-mcpu=native','-ffp-contract=off','-fno-math-errno',
    str(source),'-lm','-o',str(library)],check=True,capture_output=True,text=True)
  lib=ctypes.CDLL(str(library))
  ptr=ctypes.c_void_p
  lib.pack.argtypes=[ctypes.c_int]*4+[ptr]*5
  lib.pack.restype=ctypes.c_int
  lib.normalize.argtypes=[ctypes.c_int]*4+[ptr]*2
  lib.normalize.restype=ctypes.c_int
  lib.unpack_result.argtypes=[ctypes.c_int]*3+[ptr]*2
  lib.unpack_result.restype=ctypes.c_int
  lib._test_tempdir=directory
  return lib

def ptr(array):return ctypes.c_void_p(array.ctypes.data)

def group_heads(logical):
  result=np.empty_like(logical)
  result[Q_TO_GROUPED]=logical
  return result

def tiled_offset(capacity:int,width:int,row:int,column:int,lanes:int)->int:
  tile=row//64
  local=row%64
  m=min(64,capacity-tile*64)
  return tile*64*width+(column//lanes)*m*lanes+local*lanes+column%lanes

def tiled_layout(logical:np.ndarray,capacity:int,width:int,lanes:int)->np.ndarray:
  physical=np.zeros_like(logical)
  for head in range(logical.shape[0]):
    src=logical[head].reshape(-1)
    dst=physical[head].reshape(-1)
    for row in range(capacity):
      for col in range(width):dst[tiled_offset(capacity,width,row,col,lanes)]=src[row*width+col]
  return physical

def query_layout(logical:np.ndarray,capacity:int,row_major:bool)->np.ndarray:
  grouped=group_heads(logical)
  if row_major:return grouped.copy()
  return tiled_layout(grouped,capacity,DIM,8)

def result_layout(logical:np.ndarray,capacity:int,row_major:bool)->np.ndarray:
  grouped=group_heads(logical)
  if row_major:return grouped.copy()
  return tiled_layout(grouped,capacity,DIM,4)

def score_layout(logical:np.ndarray,capacity:int,padded:int,row_major:bool,lanes:int)->np.ndarray:
  grouped=group_heads(logical)
  if row_major:return grouped.copy()
  return tiled_layout(grouped,capacity,padded,lanes)

def cache_pack_oracle(cache:np.ndarray,total:int,padded:int)->tuple[np.ndarray,np.ndarray]:
  key=np.zeros((KV_HEADS,padded,DIM),np.float16)
  value=np.zeros_like(key)
  key[:,:total]=cache[0,0,:,:total]
  value[:,:total]=cache[1,0,:,:total]
  packed_key=np.zeros_like(key)
  packed_value=np.zeros_like(value)
  for kv in range(KV_HEADS):
    for t in range(padded):
      for c in range(DIM):
        ki=(t//16)*DIM*16+(c//32)*512+(t%16)*32+c%32
        vi=(c//16)*padded*16+(t//32)*512+(c%16)*32+t%32
        packed_key[kv].reshape(-1)[ki]=key[kv,t,c]
        packed_value[kv].reshape(-1)[vi]=value[kv,t,c]
  return packed_key,packed_value

def probabilities_oracle(scores:np.ndarray,capacity:int,active:int,total:int,padded:int)->np.ndarray:
  probabilities=np.zeros((HEADS,capacity,padded),np.float16)
  for head in range(HEADS):
    for row in range(active):
      visible=total-active+row+1
      values=scores[head,row,:visible].astype(np.float32)/np.float32(np.sqrt(DIM))
      exp=np.exp(values-values.max())
      probabilities[head,row,:visible]=(exp/exp.sum()).astype(np.float16)
  return group_heads(probabilities)

def unpack_oracle(raw:np.ndarray,active:int,capacity:int)->np.ndarray:
  expected=np.zeros((HEADS,capacity,DIM),np.float32)
  expected[:,:active]=raw[:,:active]
  return expected

class TestTinyLlamaRockchipAttentionCPU(unittest.TestCase):
  def test_grouped_query_key_value_and_result_layouts(self):
    # Exercise both the primary row-major/token-major fast path and the existing
    # blocked fallback. Active-row tails cross the 64-row physical tile boundary.
    layouts=((True,True),(False,False))
    cases=((17,17,128,128),(65,17,190,256),(65,65,512,512),(128,65,128,128))
    for row_major,token_major in layouts:
      lib=compile_helper(row_major,token_major)
      rng=np.random.default_rng(20261031+int(row_major))
      for capacity,active,total,padded in cases:
        query=rng.normal(0,.25,(HEADS,capacity,DIM)).astype(np.float32)
        query[:,active:]=np.nan
        cache=rng.normal(0,.25,(2,1,KV_HEADS,CONTEXT,DIM)).astype(np.float16)
        cache[:,:,:,total:]=np.nan
        packed_query=np.full((HEADS,capacity,DIM),np.nan,np.float16)
        packed_key=np.empty((KV_HEADS,padded,DIM),np.float16)
        packed_value=np.empty_like(packed_key)
        self.assertEqual(lib.pack(capacity,active,total,padded,ptr(query),ptr(cache),
          ptr(packed_query),ptr(packed_key),ptr(packed_value)),0)

        expected_query=np.zeros((HEADS,capacity,DIM),np.float32)
        expected_query[:,:active]=query[:,:active]
        expected_query=query_layout(expected_query,capacity,row_major).astype(np.float16)
        np.testing.assert_array_equal(packed_query.view(np.uint16),expected_query.view(np.uint16))
        expected_key,expected_value=cache_pack_oracle(cache,total,padded)
        np.testing.assert_array_equal(packed_key.view(np.uint16),expected_key.view(np.uint16))
        np.testing.assert_array_equal(packed_value.view(np.uint16),expected_value.view(np.uint16))

        # Poison noncausal score cells, including inactive query rows. normalize
        # must read only total-active+row+1 positions and zero every other lane.
        logical_scores=rng.normal(0,.4,(HEADS,capacity,padded)).astype(np.float32)
        for row in range(capacity):
          visible=total-active+row+1 if row<active else 0
          logical_scores[:,row,visible:]=np.nan
        raw_scores=score_layout(logical_scores,capacity,padded,row_major,4)
        probabilities=np.full((HEADS,capacity,padded),np.nan,np.float16)
        self.assertEqual(lib.normalize(capacity,active,total,padded,ptr(raw_scores),ptr(probabilities)),0)
        expected_prob=probabilities_oracle(logical_scores,capacity,active,total,padded)
        if not row_major:expected_prob=tiled_layout(expected_prob,capacity,padded,8)
        np.testing.assert_allclose(probabilities,expected_prob,rtol=2e-3,atol=2e-5)

        raw_result=rng.normal(0,.3,(HEADS,capacity,DIM)).astype(np.float32)
        raw_result[:,active:]=np.nan
        packed_result=result_layout(raw_result,capacity,row_major)
        output_shape=(capacity,HEADS,DIM) if token_major else (HEADS,capacity,DIM)
        output=np.full(output_shape,np.nan,np.float32)
        self.assertEqual(lib.unpack_result(capacity,active,padded,ptr(packed_result),ptr(output)),0)
        if token_major:output=output.transpose(1,0,2)
        expected_output=unpack_oracle(raw_result,active,capacity)
        np.testing.assert_array_equal(output.view(np.uint32),expected_output.view(np.uint32))
        self.assertTrue(np.all(output[:,active:]==0))

  def test_query_group_order_is_eight_queries_per_kv_head(self):
    self.assertEqual(GROUP,8)
    expected=np.asarray([kv*GROUP+group for group in range(GROUP) for kv in range(KV_HEADS)])
    np.testing.assert_array_equal(np.argsort(Q_TO_GROUPED),expected)
    # Distinct head markers verify that packed query rows are ordered by
    # (query-within-group, KV head), while KV tensors retain four heads.
    capacity,active,total,padded=17,17,128,128
    query=np.empty((HEADS,capacity,DIM),np.float32)
    for h in range(HEADS):query[h]=np.float32(h+1)
    cache=np.ones((2,1,KV_HEADS,CONTEXT,DIM),np.float16)
    qa=np.empty((HEADS,capacity,DIM),np.float16)
    ka=np.empty((KV_HEADS,padded,DIM),np.float16)
    va=np.empty_like(ka)
    lib=compile_helper(True,True)
    self.assertEqual(lib.pack(capacity,active,total,padded,ptr(query),ptr(cache),ptr(qa),ptr(ka),ptr(va)),0)
    np.testing.assert_array_equal(qa[:,0,0],(expected+1).astype(np.float16))

  def test_visible_nonfinite_scores_are_rejected(self):
    lib=compile_helper(True,True)
    capacity,active,total,padded=17,17,128,128
    for invalid in (np.nan,np.inf,-np.inf):
      logical=np.zeros((HEADS,capacity,padded),np.float32)
      logical[3,2,0]=invalid
      packed=score_layout(logical,capacity,padded,True,4)
      probability=np.empty((HEADS,capacity,padded),np.float16)
      self.assertEqual(lib.normalize(capacity,active,total,padded,ptr(packed),ptr(probability)),11)
