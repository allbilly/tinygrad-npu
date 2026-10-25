"""Fetch explicitly selected public Q8 GGUF candidates from the benchmark catalog.

Pins immutable revisions, checks upstream LFS byte counts and SHA256, resumes only
the same artifact, and retains per-model progress. It never accepts model code.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, http.client, json, shutil, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path


def segment(base:str, start:int, end:int, size:int, path:Path) -> Path:
  failures = 0
  while True:
    offset = path.stat().st_size if path.exists() else 0
    if offset==end-start+1:return path
    if offset>end-start+1:raise ValueError('Cached segment exceeds its range')
    suffix = urllib.parse.urlencode({'download':'true','onpu':time.time_ns()})
    position = start+offset
    request = urllib.request.Request(base+'?'+suffix,headers={'Range':f'bytes={position}-{end}'})
    try:
      with urllib.request.urlopen(request,timeout=120) as response:
        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {position}-{end}/{size}':
          raise ValueError('Server did not honor the requested byte range')
        with path.open('ab' if path.exists() else 'xb') as output:
          while chunk:=response.read(4<<20):
            if position+len(chunk)>end+1:raise ValueError('Download exceeds requested range')
            output.write(chunk)
            position+=len(chunk)
      if position!=end+1:raise EOFError('Connection ended before the requested range completed')
      return path
    except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.IncompleteRead,EOFError):
      progressed = path.exists() and path.stat().st_size>offset
      failures = 0 if progressed else failures+1
      if failures>=3:raise
      time.sleep(max(1,failures))


def fetch(row:dict, root:Path, connections:int=4, range_bytes:int=32<<20) -> dict:
  probe = row['gguf_probe']
  files = probe['files']
  if len(files)!=1:raise ValueError('Select exactly one unambiguous Q8 file')
  source = {'model':row['model'],'size':row['size'],'source_model':row['checkpoint_candidate'],
            'repo':probe['resolved_repo'],'revision':probe['revision']}
  return fetch_file(files[0],root/(row['model']+'-'+row['size']).replace('/','_'),source,connections,range_bytes)


def fetch_file(item:dict, directory:Path, source_info:dict, connections:int=4, range_bytes:int=32<<20, *,
               metadata_name:str='download.json') -> dict:
  """Fetch one pinned LFS artifact; separate metadata names allow sharded snapshots."""
  if Path(metadata_name).name!=metadata_name:raise ValueError('Metadata must be a filename')
  expected,size = item['lfs']['sha256'],item['size']
  filename = Path(item['rfilename']).name
  directory.mkdir(parents=True,exist_ok=True)
  target = directory/filename
  partial = target.with_suffix('.partial')
  parts = directory/(filename+'.parts')
  metadata = directory/metadata_name
  identity = {**source_info,'filename':item['rfilename'],
              'expected_sha256':expected,'expected_bytes':size,'checkpoint':str(target)}
  if metadata.exists():
    old = json.loads(metadata.read_text())
    if any(old.get(k)!=v for k,v in identity.items()):raise ValueError('Existing download belongs to a different artifact')
  elif partial.exists() or target.exists():raise ValueError('Existing checkpoint has no recorded artifact identity')
  result = {**identity,'complete':False,'connections':connections}
  def save():
    pending = metadata.with_suffix('.tmp')
    pending.write_text(json.dumps(result,indent=2)+'\n')
    pending.replace(metadata)
  save()
  try:
    digest = hashlib.sha256()
    count = 0
    existing = target if target.exists() else partial
    if existing.exists():
      with existing.open('rb') as source:
        for chunk in iter(lambda:source.read(4<<20),b''):
          digest.update(chunk)
          count+=len(chunk)
    if target.exists():
      if count!=size or digest.hexdigest()!=expected:raise ValueError('Existing checkpoint checksum differs')
    else:
      if count>size:raise ValueError('Partial checkpoint exceeds expected size')
      parts.mkdir(exist_ok=True)
      base = f"https://huggingface.co/{identity['repo']}/resolve/{identity['revision']}/{urllib.parse.quote(item['rfilename'])}"
      stamp = time.monotonic()
      with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        while count<size:
          width = min(range_bytes,(size-count+connections-1)//connections)
          spans = [(start,min(size,start+width)-1) for start in range(count,min(size,count+width*connections),width)]
          paths = [parts/f'{start}-{end}.part' for start,end in spans]
          futures = [pool.submit(segment,base,start,end,size,path) for (start,end),path in zip(spans,paths)]
          pending = set(futures)
          while pending:
            _,pending = concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
            if time.monotonic()-stamp>=30:
              result['downloaded_bytes']=count+sum(p.stat().st_size for p in paths if p.exists())
              save()
              print(json.dumps({'model':identity['model'],'size':identity['size'],'filename':filename,
                                'bytes':result['downloaded_bytes'],'total':size}),flush=True)
              stamp=time.monotonic()
          # Commit only contiguous verified-length segments; an interrupted append
          # leaves a valid prefix that the next invocation can safely resume.
          for future in futures:
            path = future.result()
            with path.open('rb') as source,partial.open('ab') as output:
              for chunk in iter(lambda:source.read(4<<20),b''):
                output.write(chunk)
                digest.update(chunk)
                count+=len(chunk)
            path.unlink()
      if count!=size or digest.hexdigest()!=expected:raise ValueError('Downloaded checkpoint checksum differs')
      partial.rename(target)
    result.update(complete=True,downloaded_bytes=count,sha256=digest.hexdigest())
    if parts.exists():
      for path in parts.glob('*.part'):
        bounds = path.stem.split('-')
        if len(bounds)==2 and all(v.isdigit() for v in bounds) and 0<=int(bounds[0])<=int(bounds[1])<size:path.unlink()
      if not any(parts.iterdir()):parts.rmdir()
  except Exception as exc:result.update(error=f'{type(exc).__name__}: {exc}',downloaded_bytes=count)
  save()
  print(json.dumps({'model':identity['model'],'size':identity['size'],'filename':filename,
                    'complete':result['complete'],'error':result.get('error')}),flush=True)
  return result


def fetch_selected(selected:list[dict], root:Path, workers:int=2, connections:int=4, rounds:int=1, retry_delay:float=30) -> dict:
  if rounds<1 or retry_delay<0:raise ValueError('Need positive retry rounds and nonnegative delay')
  keys = [row['model']+'/'+row['size'] for row in selected]
  if not keys or len(keys)!=len(set(keys)):raise ValueError('Need distinct selected models')
  selection = hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()[:12]
  manifest = root/f'downloads-{selection}.json'
  results:dict[str,dict] = {}
  report:dict = {}
  for attempt in range(rounds):
    pending = [row for row,key in zip(selected,keys) if not results.get(key,{}).get('complete')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
      for result in pool.map(lambda row:fetch(row,root,connections),pending):
        results[result['model']+'/'+result['size']] = result
    report = {'requested_models':keys,'models':[results[key] for key in keys],
              'complete':all(results[key]['complete'] for key in keys),'round':attempt+1,'maximum_rounds':rounds}
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(manifest)
    if report['complete']:break
    if attempt+1<rounds:
      print(json.dumps({'event':'retry_pass','round':attempt+2,'delay_seconds':retry_delay,
                        'pending':[key for key in keys if not results[key]['complete']]}),flush=True)
      time.sleep(retry_delay)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--catalog',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--only',action='append',required=True,help='Explicit model/size key; repeat to add checkpoints')
  parser.add_argument('--workers',type=int,choices=(1,2,3),default=2)
  parser.add_argument('--connections',type=int,choices=(1,2,4),default=4)
  parser.add_argument('--retry-rounds',type=int,default=1,help='Bounded passes; only failed model downloads repeat')
  parser.add_argument('--retry-delay',type=float,default=30,help='Seconds between failed passes')
  args = parser.parse_args()
  if args.retry_rounds<1 or args.retry_delay<0:parser.error('Need positive retry rounds and nonnegative delay')
  catalog = json.loads(args.catalog.read_text())
  by_key = {row['model']+'/'+row['size']:row for row in catalog['models']}
  if len(args.only)!=len(set(args.only)) or any(key not in by_key for key in args.only):parser.error('Unknown or repeated model selection')
  selected = [by_key[key] for key in args.only]
  if any(len(row.get('gguf_probe',{}).get('files',[]))!=1 for row in selected):parser.error('Missing or ambiguous Q8 checkpoint')
  if any(row['gguf_probe'].get('gated') for row in selected):parser.error('Only public checkpoint downloads are supported')
  args.output.mkdir(parents=True,exist_ok=True)
  size = sum(row['gguf_probe']['files'][0]['size'] for row in selected)
  if size+8*1024**3>shutil.disk_usage(args.output).free:parser.error('Insufficient disk space plus 8GiB reserve')
  report = fetch_selected(selected,args.output,args.workers,args.connections,args.retry_rounds,args.retry_delay)
  if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
