"""Record measured model baselines and update the original-format RK3588 table.

Published results and local baselines are distinct measurements. A faster local
number alone never certifies comparable quality or a matched-reference win.
"""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from pathlib import Path


def summarize(report:dict) -> dict:
  if not report.get('complete'):raise ValueError('Incomplete benchmark')
  rows = report.get('trials',[])
  warmups = [r for r in rows if r.get('warmup') is True]
  measured = [r for r in rows if r.get('warmup') is False]
  if len(warmups)<3 or len(measured)<3:raise ValueError('Need three actual warmups and three measured runs')
  if report.get('context',0)<192:raise ValueError('Context cannot hold the requested workload')
  if len(report.get('input_ids',[]))!=128:raise ValueError('Expected 128 input tokens')
  for row in rows:
    if row.get('prompt_tokens')!=128 or len(row.get('generated_ids',[]))!=64:raise ValueError('Different input/output length')
    if any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<=0 for k in ('first_token_ms','decode_tps')):
      raise ValueError('Invalid timing')
  if any(row['generated_ids']!=measured[0]['generated_ids'] for row in measured):raise ValueError('Greedy output changed across measured runs')
  if report.get('device') not in ('CPU','CL'):raise ValueError('Unknown measured device')
  if not report.get('checkpoint_sha256') or not report.get('source_sha256'):raise ValueError('Missing weight/source provenance')
  memory = [r.get('peak_rss_bytes') for r in measured]
  precision = 'INT8' if report.get('precision','').startswith('INT8') else 'FP16'
  return {'context':report['context'],'input_tokens':128,'output_tokens':64,'warmups':len(warmups),'runs':len(measured),
          'ttft_ms':statistics.median(r['first_token_ms'] for r in measured),
          'prefill_proxy_tps':statistics.median(128000/r['first_token_ms'] for r in measured),
          'decode_tps':statistics.median(r['decode_tps'] for r in measured),
          'peak_rss_mib':max(memory)/1024**2 if all(isinstance(v,int) and v>0 for v in memory) else None,
          'device':report['device'],'precision':precision,'placement':report.get('placement'),
          'gpu_kernels':report.get('gpu_kernels'),'checkpoint_sha256':report['checkpoint_sha256'],
          'input_sha256':hashlib.sha256(json.dumps(report['input_ids']).encode()).hexdigest(),
          'qualified_win':False,'quality_status':'not qualified against matched level-0 RKLLM'}


def pair(reference:float|None, local:float|None, *, lower:bool=False) -> str:
  if reference is None:return '— / '+(f'{local:.2f}' if local is not None else '—')
  if local is None:return f'{reference:.2f} / —'
  a,b = f'{reference:.2f}',f'{local:.2f}'
  if local!=reference:
    if (local<reference)==lower:b=f'**{b}**'
    else:a=f'**{a}**'
  return a+' / '+b


def render(coverage:dict, results:dict, *, additional:bool=False) -> str:
  rows = ['| Model | Model Size | Dtype | Seqlen | New_tokens | TTFT(ms) | Tokens/s | memory(MB) |',
          '| :---------- | :--------: | :---: | :----: | :--------: | :------: | :------: | :--------: |']
  seen = set()
  models = coverage['models']+coverage.get('benchmark_additions',[])
  for model in models:
    key = model['model']+'/'+model['size']
    if key in seen:raise ValueError('Duplicate model/size in coverage')
    seen.add(key)
    local = results.get(key,{})
    if local and (local['input_tokens']!=128 or local['output_tokens']!=64):raise ValueError('Wrong workload in local results')
  if set(results)-seen:raise ValueError('Local result does not match a cataloged model/size')
  for model in (coverage.get('benchmark_additions',[]) if additional else coverage['models']):
    local = results.get(model['model']+'/'+model['size'],{})
    ttft = pair(model['published_ttft_ms'],local.get('ttft_ms'),lower=True)
    decode = pair(model['published_decode_tps'],local.get('decode_tps'))
    memory = model['published_memory_mb']
    mem = (f'{memory:.2f}' if memory is not None else '—')+' / '+(f"{local['peak_rss_mib']:.2f}*" if local.get('peak_rss_mib') else '—')
    dtype = (model['dtype'] or '—')+' / '+local.get('precision','—')
    rows.append(f"| {model['model']} | {model['size']} | {dtype} | 128 | 64 | {ttft} | {decode} | {mem} |")
  return '\n'.join(rows)


def render_prefill(coverage:dict, results:dict) -> str:
  render(coverage,results)  # Apply the same identity/workload checks as the main table.
  rows = ['| Model | Model Size | Prefill proxy(tok/s), RKLLM / oNPU |',
          '| :--- | :---: | ---: |']
  for model in coverage['models']+coverage.get('benchmark_additions',[]):
    local = results.get(model['model']+'/'+model['size'],{})
    reference = 128000/model['published_ttft_ms'] if model['published_ttft_ms'] is not None else None
    measured = 128000/local['ttft_ms'] if local else None
    rows.append(f"| {model['model']} | {model['size']} | {pair(reference,measured)} |")
  return '\n'.join(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for key in ('coverage','results','benchmark','report'):parser.add_argument('--'+key,type=Path,required=True)
  parser.add_argument('--model',required=True)
  parser.add_argument('--size',required=True)
  args = parser.parse_args()
  coverage = json.loads(args.coverage.read_text())
  report = json.loads(args.report.read_text())
  entry = summarize(report)
  entry.update(report=str(args.report.resolve()),report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest())
  results = json.loads(args.results.read_text()) if args.results.exists() else {}
  results[args.model+'/'+args.size] = entry
  table = render(coverage,results)
  text = args.benchmark.read_text()
  begin = text.index('| Model | Model Size |')
  end = begin
  while end<len(text) and text[end]=='|':end=text.index('\n',end)+1
  text = text[:begin]+table+'\n'+text[end:]
  if coverage.get('benchmark_additions'):
    start_marker,end_marker = '<!-- onpu-additional-benchmarks:start -->','<!-- onpu-additional-benchmarks:end -->'
    begin,end = text.index(start_marker)+len(start_marker),text.index(end_marker)
    text = text[:begin]+'\n'+render(coverage,results,additional=True)+'\n'+text[end:]
  start_marker,end_marker = '<!-- onpu-prefill:start -->','<!-- onpu-prefill:end -->'
  if start_marker in text:
    begin,end = text.index(start_marker)+len(start_marker),text.index(end_marker)
    text = text[:begin]+'\n'+render_prefill(coverage,results)+'\n'+text[end:]
  args.benchmark.write_text(text)
  args.results.write_text(json.dumps(results,indent=2)+'\n')
  print(json.dumps({'model':args.model,'size':args.size,**entry}))


if __name__=='__main__':main()
