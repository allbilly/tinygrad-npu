"""CPU-only CLI preparation must run without creating a Rockchip device."""
import json, sys
from tinygrad import Device
from extra.rockchip import benchmark_inference
from test.models.test_minicpm3 import TestMiniCPM3 as MiniReference


def run_benchmark(model, *, prepare, tmp_path, monkeypatch):
  checkpoint=tmp_path/'fixture.gguf'
  checkpoint.write_bytes(b'fixture checkpoint bytes; model_override supplies weights')
  input_path=tmp_path/'input.ids'
  input_path.write_text('3 7 2\n')
  report_path=tmp_path/('prepared.json' if prepare else 'control.json')
  argv=['benchmark_inference','--checkpoint',str(checkpoint),'--input',str(input_path),
    '--report',str(report_path),'--device','CPU','--prefill','off','--decode','off',
    '--precision','fp16','--context','16','--tokens','2','--warmups','1','--runs','1']
  if prepare:argv.extend(['--cpu-mla-weights','--cpu-mla-attention'])
  monkeypatch.setattr(sys,'argv',argv)
  benchmark_inference.main(model_override=model)
  return json.loads(report_path.read_text())


def test_cpu_mla_prepare_flag_works_without_npu_phases(tmp_path,monkeypatch):
  fixture=MiniReference()
  weights=fixture.fixture()
  monkeypatch.delenv('ONPU_BENCHMARK_CHECKPOINT_SHA256',raising=False)
  before_rockchip={name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}
  control=run_benchmark(fixture.load(weights),prepare=False,tmp_path=tmp_path,monkeypatch=monkeypatch)
  prepared=run_benchmark(fixture.load(weights),prepare=True,tmp_path=tmp_path,monkeypatch=monkeypatch)
  assert control['complete'] is True and prepared['complete'] is True
  assert control['placement'] is None and prepared['placement'] is None
  assert control['cpu_mla_weights_prepared']==0 and prepared['cpu_mla_weights_prepared']==4
  assert control['cpu_mla_attention_blocks']==0 and prepared['cpu_mla_attention_blocks']==2
  control_ids=[trial['generated_ids'] for trial in control['trials'] if not trial['warmup']]
  prepared_ids=[trial['generated_ids'] for trial in prepared['trials'] if not trial['warmup']]
  assert control_ids==prepared_ids and len(control_ids)==len(prepared_ids)==1
  assert {name for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'}==before_rockchip
