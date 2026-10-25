"""Export and verify self-describing RKLLM references for matched benchmarks.

The published RK3588 recipe requires optimization_level=0. Level 1 is available
only to reproduce older exports; it is rejected by the qualification validator.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path


def file_hash(path:Path) -> str:
  # Converter images use Python 3.10, before hashlib.file_digest was introduced.
  checksum = hashlib.sha256()
  with path.open('rb') as source:
    for chunk in iter(lambda:source.read(8<<20),b''):checksum.update(chunk)
  return checksum.hexdigest()


def provenance_path(artifact:Path) -> Path:return artifact.with_suffix('.provenance.json')


def export_reference(model:Path, dataset:Path, artifact:Path, *, context:int=512, optimization_level:int=0,
                     source_checkpoint:Path|None=None) -> dict:
  if context<32 or context%32:raise ValueError('Context must be a positive multiple of 32')
  if optimization_level not in (0,1):raise ValueError('Expected level 0, or explicit legacy level 1')
  model,dataset,artifact = model.resolve(),dataset.resolve(),artifact.resolve()
  metadata_path = provenance_path(artifact)
  pending = artifact.with_name(artifact.stem+'.partial.rkllm')
  if any(path.exists() for path in (artifact,metadata_path,pending)):
    raise FileExistsError('Reference output already exists; use a new artifact path')
  weights = sorted(model.glob('*.safetensors'))
  if not weights:raise ValueError('Reference model has no safetensor weights')
  config = json.loads((model/'config.json').read_text())
  calibration = json.loads(dataset.read_text())
  calibration_files = {str(dataset):file_hash(dataset)}
  for row in calibration if isinstance(calibration,list) else ():
    if isinstance(row,dict) and isinstance(row.get('sample'),str):
      sample = dataset.parent/row['sample']
      calibration_files[str(sample)] = file_hash(sample)
  build = {'do_quantization':True,'optimization_level':optimization_level,'quantized_dtype':'w8a8',
           'quantized_algorithm':'normal','target_platform':'rk3588','num_npu_core':3,'dataset':str(dataset),'max_context':context}
  metadata = {'schema_version':1,'optimization_level':optimization_level,'max_context':context,'build_parameters':build,
              'model_config':config,'model_files_sha256':{p.name:file_hash(p) for p in weights},
              'config_sha256':file_hash(model/'config.json'),'calibration_files_sha256':calibration_files,
              'exporter_sha256':file_hash(Path(__file__))}
  if source_checkpoint is not None:metadata['source_sha256'] = file_hash(source_checkpoint)
  from importlib.metadata import version
  metadata['toolkit'] = version('rkllm-toolkit')
  artifact.parent.mkdir(parents=True,exist_ok=True)
  previous = Path.cwd()
  # The SDK creates temporary conversion files in its working directory.
  # Resolve all inputs first and keep those files beside the writable output.
  with tempfile.TemporaryDirectory(prefix='rkllm-export-',dir=artifact.parent) as workspace:
    try:
      os.chdir(workspace)
      from rkllm.api import RKLLM
      llm = RKLLM()
      try:
        if llm.load_huggingface(model=str(model),device='cpu',dtype='float16'):
          raise RuntimeError('RKLLM reference loading failed')
        if llm.build(**build):raise RuntimeError('RKLLM reference build failed')
        if llm.export_rkllm(str(pending)):raise RuntimeError('RKLLM reference export failed')
        metadata.update(artifact_sha256=file_hash(pending),artifact=str(artifact))
        pending.rename(artifact)
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
      finally:
        # Only the new, previously nonexistent partial file belongs to this export.
        if pending.exists():pending.unlink()
        if callable(getattr(llm,'release',None)):llm.release()
    finally:os.chdir(previous)
  return metadata


def validate_reference(artifact:Path, metadata_path:Path|None=None, *, context:int|None=None) -> dict:
  metadata = json.loads((metadata_path or provenance_path(artifact)).read_text())
  build = metadata.get('build_parameters',{})
  if metadata.get('schema_version')!=1 or metadata.get('optimization_level')!=0 or build.get('optimization_level')!=0:
    raise ValueError('Published-recipe qualification requires recorded optimization_level=0')
  required = {'do_quantization':True,'quantized_dtype':'w8a8','target_platform':'rk3588','num_npu_core':3}
  if any(build.get(key)!=value for key,value in required.items()):raise ValueError('Reference build differs from the RK3588 W8A8 recipe')
  if context is not None and (metadata.get('max_context')!=context or build.get('max_context')!=context):
    raise ValueError('Reference and candidate allocated contexts differ')
  if metadata.get('artifact_sha256')!=file_hash(artifact):raise ValueError('Reference artifact checksum mismatch')
  return metadata


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model',type=Path,required=True)
  parser.add_argument('--dataset',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--optimization-level',type=int,choices=(0,1),default=0)
  args = parser.parse_args()
  print(json.dumps(export_reference(args.model,args.dataset,args.output,context=args.context,optimization_level=args.optimization_level)),flush=True)


if __name__=='__main__':main()
