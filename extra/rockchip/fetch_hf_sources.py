"""Download a pinned public HF weight snapshot for models without a public GGUF.

Verifies LFS SHA256 and Git blob hashes as declared by the snapshot. Resumes each
shard independently, retries failed shards, and never executes downloaded code.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, shutil, time, urllib.request
from pathlib import Path
from extra.rockchip.fetch_benchmark_checkpoints import fetch_file


def git_blob(data:bytes) -> str:
  return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def small_file(repo:str, revision:str, item:dict, directory:Path) -> dict:
  name = item['rfilename']
  if Path(name).name!=name or item['size']>16<<20:raise ValueError('Expected a small root-level snapshot file')
  path = directory/name
  if path.exists():data = path.read_bytes()
  else:
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    for attempt in range(3):
      try:
        with urllib.request.urlopen(url,timeout=60) as response:data = response.read(item['size']+1)
        break
      except (OSError,TimeoutError):
        if attempt==2:raise
        time.sleep(attempt+1)
  digest = hashlib.sha256(data).hexdigest()
  verified = digest==item['lfs']['sha256'] if item.get('lfs') else git_blob(data)==item['blobId']
  if len(data)!=item['size'] or not verified:raise ValueError('Small snapshot file checksum differs: '+name)
  if not path.exists():
    temporary = path.with_name(name+'.download.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)
  return {'filename':name,'bytes':len(data),'git_blob':item['blobId'],'sha256':digest,
          'verification':'lfs_sha256' if item.get('lfs') else 'git_blob','complete':True}


def weight_files(metadata:dict, directory:Path) -> list[dict]:
  items = {item['rfilename']:item for item in metadata['siblings']}
  index = next((name for name in ('model.safetensors.index.json','pytorch_model.bin.index.json') if name in items),None)
  if index is not None:
    names = sorted(set(json.loads((directory/index).read_text())['weight_map'].values()))
  else:names = [next((name for name in ('model.safetensors','pytorch_model.bin') if name in items),'')]
  if not names or any(name not in items or Path(name).name!=name or not items[name].get('lfs') for name in names):
    raise ValueError('Need a complete indexed or single-file LFS weight snapshot')
  return [items[name] for name in names]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('repo','revision','model','size'):parser.add_argument('--'+name,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--metadata',type=Path,help='Previously fetched pinned HF API metadata')
  parser.add_argument('--workers',type=int,choices=(1,2),default=2)
  parser.add_argument('--connections',type=int,choices=(1,2,4),default=4)
  parser.add_argument('--retry-rounds',type=int,default=16)
  args = parser.parse_args()
  if len(args.revision)!=40 or any(c not in '0123456789abcdef' for c in args.revision) or args.retry_rounds<1:
    parser.error('Use an immutable revision and positive retry count')
  source = {'model':args.model,'size':args.size,'source_model':args.repo,'repo':args.repo,'revision':args.revision}
  args.output.mkdir(parents=True,exist_ok=True)
  manifest = args.output/'source-manifest.json'
  if manifest.exists():
    old = json.loads(manifest.read_text())
    if any(old.get(key)!=value for key,value in source.items()):raise ValueError('Existing snapshot identity differs')
  report = {**source,'complete':False,'format':'original HF weights','files':[]}
  def save():
    pending = manifest.with_suffix('.tmp')
    pending.write_text(json.dumps(report,indent=2)+'\n')
    pending.replace(manifest)
  save()
  try:
    url = f'https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true'
    if args.metadata:metadata = json.loads(args.metadata.read_text())
    else:
      with urllib.request.urlopen(url,timeout=60) as response:metadata = json.load(response)
    if metadata['sha']!=args.revision or metadata['id']!=args.repo or metadata.get('gated'):
      raise ValueError('Expected the pinned public snapshot')
    wanted = {'config.json','tokenizer.json','tokenizer.model','tokenizer_config.json','generation_config.json',
              'model.safetensors.index.json','pytorch_model.bin.index.json'}
    small = [item for item in metadata['siblings'] if item['rfilename'] in wanted or item['rfilename'].endswith('.py')]
    for item in small:
      report['files'].append(small_file(args.repo,args.revision,item,args.output))
      save()
    weights = weight_files(metadata,args.output)
    if sum(item['size'] for item in weights)+8*1024**3>shutil.disk_usage(args.output).free:raise ValueError('Insufficient disk reserve')
    results:dict[str,dict] = {}
    for attempt in range(args.retry_rounds):
      pending = [item for item in weights if not results.get(item['rfilename'],{}).get('complete')]
      def fetch(item):return fetch_file(item,args.output,source,args.connections,metadata_name=Path(item['rfilename']).name+'.download.json')
      with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(fetch,pending):results[result['filename']] = result
      report.update(round=attempt+1,weights=[results[item['rfilename']] for item in weights],
                    complete=all(results[item['rfilename']]['complete'] for item in weights))
      save()
      if report['complete']:break
      if attempt+1<args.retry_rounds:time.sleep(30)
  except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise
  if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
