"""CPU-only smoke check that benchmark chunk selection sees the requested hook."""
import json, sys
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig
from extra.rockchip import benchmark_inference


def test_recurrent_benchmark_installs_hook_before_choosing_prefill_chunk(tmp_path,monkeypatch):
  ssm=SSMConfig(2,8,1,1,8)
  model=Transformer(TransformerConfig(1,16,32,2,1,1e-5,32,8,10000.,8,8,
    max_context=32,ssm=ssm,ssm_layers=(True,)))
  model.gguf_loading={'scope':'CPU smoke fixture with an injected model; no checkpoint loading'}
  checkpoint=tmp_path/'fixture.gguf'
  checkpoint.write_bytes(b'injected CPU smoke model')
  prompt=tmp_path/'input.txt'
  prompt.write_text('1 2 3 4 5 6 7 8\n')
  report=tmp_path/'report.json'
  monkeypatch.delenv('ONPU_BENCHMARK_CHECKPOINT_SHA256',raising=False)
  monkeypatch.setattr(sys,'argv',['benchmark_inference','--checkpoint',str(checkpoint),'--input',str(prompt),
    '--report',str(report),'--device','CPU','--prefill','off','--decode','off','--context','32',
    '--tokens','2','--warmups','1','--runs','1','--chunk-size','8','--cpu-recurrent-scan'])
  benchmark_inference.main(model_override=model)
  result=json.loads(report.read_text())
  assert result['complete'] is True
  assert result['cpu_recurrent_scan'] is True and result['cpu_recurrent_blocks']==1
  assert result['requested_chunk_size']==result['chunk_size']==8
  assert result['placement'] is None  # This smoke test never opens the NPU.
  assert len(result['trials'])==2
  assert 'tinygrad/llm/kernels/rockchip_recurrent.c' in result['source_sha256']
