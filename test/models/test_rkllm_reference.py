import hashlib, importlib.metadata, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from extra.rockchip.rkllm_reference import export_reference, provenance_path, validate_reference


@pytest.fixture
def converter(tmp_path,monkeypatch):
  model = tmp_path/'hf'
  model.mkdir()
  (model/'model.safetensors').write_bytes(b'shared checkpoint')
  (model/'config.json').write_text(json.dumps({'architectures':['LlamaForCausalLM']}))
  dataset = tmp_path/'calibration.json'
  dataset.write_text(json.dumps([{'sample':'calibration.pkl','token_nums':128}]))
  (tmp_path/'calibration.pkl').write_bytes(b'calibration embeddings')
  legacy = tmp_path/'text-w8a8.rkllm'
  legacy.write_bytes(b'legacy evidence')
  class SDK:
    calls = []
    failure = None
    released = 0
    export_cwds = []
    def load_huggingface(self,**kwargs):
      self.calls.append(('load',kwargs))
      return 0
    def build(self,**kwargs):
      self.calls.append(('build',kwargs))
      return int(self.failure=='build')
    def export_rkllm(self,path):
      self.calls.append(('export',path))
      self.export_cwds.append(Path.cwd())
      Path('sdk-temporary-output').write_bytes(b'writable workspace required')
      Path(path).write_bytes(b'new artifact')
      return int(self.failure=='export')
    def release(self):SDK.released += 1
  monkeypatch.setitem(sys.modules,'rkllm',SimpleNamespace())
  monkeypatch.setitem(sys.modules,'rkllm.api',SimpleNamespace(RKLLM=SDK))
  version = importlib.metadata.version
  monkeypatch.setattr(importlib.metadata,'version',lambda name:'1.3.1-test' if name=='rkllm-toolkit' else version(name))
  return SimpleNamespace(model=model,dataset=dataset,legacy=legacy,sdk=SDK,artifact=tmp_path/'text-w8a8-opt0-ctx512.rkllm')


def test_default_export_uses_published_recipe_and_hashes_real_inputs(converter):
  c = converter
  export_reference(c.model,c.dataset,c.artifact)
  metadata = validate_reference(c.artifact,context=512)
  build = next(value for event,value in c.sdk.calls if event=='build')
  assert build['optimization_level']==0 and build['do_quantization'] and build['quantized_dtype']=='w8a8'
  assert build['num_npu_core']==3 and build['max_context']==512
  assert metadata['model_files_sha256']['model.safetensors']==hashlib.sha256(b'shared checkpoint').hexdigest()
  assert metadata['calibration_files_sha256'][str(c.dataset.parent/'calibration.pkl')]==hashlib.sha256(b'calibration embeddings').hexdigest()
  assert c.legacy.read_bytes()==b'legacy evidence'
  assert c.sdk.released==1
  assert not c.sdk.export_cwds[0].exists()


def test_export_from_readonly_checkout_uses_writable_scratch_and_restores_cwd(converter,monkeypatch):
  c = converter
  checkout = c.artifact.parent/'readonly-checkout'
  checkout.mkdir(mode=0o555)
  monkeypatch.chdir(checkout)
  try:
    export_reference(c.model,c.dataset,c.artifact)
    assert Path.cwd()==checkout
    assert c.sdk.export_cwds[0].parent==c.artifact.parent
    assert not c.sdk.export_cwds[0].exists()
  finally:checkout.chmod(0o755)


def test_legacy_export_is_recorded_but_rejected_for_qualification(converter):
  c = converter
  output = c.artifact.with_name('legacy-opt1.rkllm')
  metadata = export_reference(c.model,c.dataset,output,optimization_level=1)
  assert metadata['optimization_level']==1
  with pytest.raises(ValueError,match='optimization_level=0'):validate_reference(output)


def test_reference_rejects_overwrite_wrong_context_and_tampering(converter):
  c = converter
  export_reference(c.model,c.dataset,c.artifact)
  calls = list(c.sdk.calls)
  with pytest.raises(FileExistsError):export_reference(c.model,c.dataset,c.artifact)
  assert c.sdk.calls==calls and c.artifact.read_bytes()==b'new artifact'
  with pytest.raises(ValueError,match='contexts differ'):validate_reference(c.artifact,context=1024)
  c.artifact.write_bytes(b'changed artifact')
  with pytest.raises(ValueError,match='checksum'):validate_reference(c.artifact)


@pytest.mark.parametrize('stage',['build','export'])
def test_failed_sdk_operation_keeps_legacy_and_removes_only_partial(converter,stage):
  c = converter
  c.sdk.failure = stage
  previous = Path.cwd()
  with pytest.raises(RuntimeError,match='failed'):export_reference(c.model,c.dataset,c.artifact)
  assert not c.artifact.exists() and not provenance_path(c.artifact).exists()
  assert not c.artifact.with_name(c.artifact.stem+'.partial.rkllm').exists()
  assert c.legacy.read_bytes()==b'legacy evidence' and c.sdk.released==1
  assert Path.cwd()==previous
