"""Whole mapped side-table lookup preserves the original segmented semantics."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor,Device,dtypes
from tinygrad.helpers import Context
from tinygrad.llm.gemma4 import SegmentedEmbedding
from tinygrad.llm.gguf import ggml_data_to_tensor,_mapped_cpu_bytes
from tinygrad.llm.kernels.rockchip_embedding import RockchipSegmentedQ8Embedding,configure_segmented


def test_whole_table_boundary_nonfinite_zero_and_fallback_guards():
  with tempfile.TemporaryDirectory(prefix='q8-side-') as temporary:
    path=Path(temporary)/'sparse.bin'
    with path.open('wb') as file:file.truncate(2495610880)
    with path.open('r+b') as file:
      for token in (0,16384,262143):
        row=np.zeros((280,34),np.uint8)
        scales=np.resize(np.array([0x8000,0,0xbc00,0x3c00,1,0x7c00,0xfc00,0x7e01],np.uint16),280)
        row[:,:2]=scales.view(np.uint8).reshape(280,2)
        row[:,2:]=np.resize(np.array([0,1,-1,127,-128],np.int8),280*32).view(np.uint8).reshape(280,32)
        file.seek(token*9520)
        file.write(row.tobytes())
    disk=Tensor(path)
    maps={}
    source=SegmentedEmbedding(262144,8960,16384)
    for i,chunk in enumerate(source.chunks):
      first=i*16384*9520
      raw=_mapped_cpu_bytes(disk[first:first+16384*9520],maps)
      chunk.weight=ggml_data_to_tensor(raw,16384*8960,8).half().reshape(16384,8960)
    candidate=RockchipSegmentedQ8Embedding(source,_mapped_cpu_bytes(disk,maps),str(path))
    ids=Tensor([[-1,0,16384,262143,262144]],device='CPU',dtype=dtypes.int32).realize()
    expected=source(ids).numpy()
    assert candidate(ids).numpy().tobytes()==expected.tobytes()
    assert candidate.fast_traces==1 and candidate.fallback_traces==0
    for mode in ('training','geometry','weight'):
      original_rows,original_weight=source.rows,source.chunks[0].weight
      try:
        if mode=='geometry':source.rows=8192
        if mode=='weight':source.chunks[0].weight=Tensor.empty(16384,8960,device='CPU',dtype=dtypes.half)
        with patch.object(SegmentedEmbedding,'__call__',return_value=Tensor.ones(1,5,8960,device='CPU')) as fallback, \
             Context(TRAINING=mode=='training'):
          result=candidate(ids)
          assert result.shape==(1,5,8960) and fallback.call_count==1
      finally:source.rows,source.chunks[0].weight=original_rows,original_weight
    model=SimpleNamespace(per_layer_token_embd=candidate)
    assert not configure_segmented(model,path,False) and model.per_layer_token_embd is source
  assert not {'CL','ROCKCHIP'}.intersection(Device._opened_devices)
