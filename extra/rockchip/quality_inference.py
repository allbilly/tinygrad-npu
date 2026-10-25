"""Teacher-forced next-token quality on a fixed WikiText-2 test subset.

This is sampled conditional perplexity, not a full corpus or instruction eval.
The candidate must stay within2% perplexity of RKLLM W8A8 on the same requests.
"""
import argparse, hashlib, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

EOS_ID=151645
CHECKPOINT=Path('/home/orangepi/qwen3-bench/matched/roofline/gpu/Qwen3-0.6B-matched-f16.gguf')
STOCK_MODEL=Path('/home/orangepi/qwen3-bench/models/qwen3-0.6b_w8a8_rk3588.rkllm')


def digest(path):
  with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sources():
  root=Path(__file__).resolve().parents[2]
  paths=[p for p in (root/'tinygrad').rglob('*') if p.is_file() and p.suffix in ('.py','.c','.h')]
  return {str(p.relative_to(root)):digest(p) for p in paths}


def check_sources(hashes):
  root=Path(__file__).resolve().parents[2]
  if any(digest(root/name)!=value for name,value in hashes.items()):raise AssertionError('Quality source changed')


def prepare(corpus,manifest,contexts):
  from tinygrad.llm.gguf import gguf_parse
  from tinygrad.llm.cli import SimpleTokenizer
  kv,_=gguf_parse(CHECKPOINT)
  assert kv['tokenizer.ggml.eos_token_id']==EOS_ID
  tokenizer=SimpleTokenizer.from_gguf_kv(kv)
  tokens=tokenizer.encode(corpus.read_text())
  rng=np.random.default_rng(20261018)
  starts=rng.choice(len(tokens)-260,size=contexts,replace=False)
  cases=[]
  for index,start in enumerate(starts):
    length=128 if index%2==0 else 256
    sample=tokens[int(start):int(start)+length+4]
    for step in range(4):
      position=0 if step==0 else length+step-1
      input_ids=sample[:length] if step==0 else sample[length+step-1:length+step]
      cases.append({'name':f'c{index:03d}-s{step}','start':position,'input_ids':input_ids,'gold':sample[length+step],
                    'prefix':sample[:length+step]})
  report={'dataset':'WikiText-2 test, PyTorch examples mirror','corpus_sha256':digest(corpus),
          'checkpoint_sha256':digest(CHECKPOINT),'contexts':contexts,'scored_tokens':len(cases),
          'sampling_seed':20261018,'context':512,'quality_gate':{'max_perplexity_ratio_vs_rkllm':1.02},'cases':cases}
  manifest.parent.mkdir(parents=True,exist_ok=True)
  manifest.write_text(json.dumps(report,indent=2)+'\n')
  manifest.with_suffix('.tokens').write_text(''.join(
    f"{c['name']} {c['start']} {c['gold']} "+' '.join(map(str,c['input_ids']))+'\n' for c in cases))
  print(json.dumps({k:report[k] for k in ('contexts','scored_tokens','corpus_sha256')}),flush=True)


def run_tinygrad(mode,manifest,directory,cpu_pool=False,fast_softmax=False):
  os.environ['DEV']='CPU'
  from tinygrad import Tensor,TinyJit,Variable
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_replay import RockchipCapturedJit
  data=json.loads(manifest.read_text())
  if digest(CHECKPOINT)!=data['checkpoint_sha256']:raise AssertionError('Checkpoint changed')
  directory.mkdir(parents=True,exist_ok=False)
  model,_=Transformer.from_gguf(CHECKPOINT,max_context=512,realize=True)
  placement=configure(model,prefill='on',decode='on',tile_columns=24576,fuse_projections=True,cpu_attention=True,
                      native_replay=False,fused_ffn=True,split_down=True,fused_qkv=True,cpu_norm=True,precision=mode,
                      cpu_pool=cpu_pool,cpu_fast_softmax=fast_softmax)
  variants={}
  hashes=sources()
  report={'complete':False,'precision':mode,'placement':placement,'manifest_sha256':digest(manifest),
          'source_sha256':hashes,'cases':[]}
  target=directory/'report.json'
  def save():target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  try:
    for case in data['cases']:
      key=len(case['input_ids'])
      if key not in variants:variants[key]=TinyJit(lambda tokens,start:model.logits(tokens,start).realize())
      jit=variants[key]
      inputs=Tensor([case['input_ids']],device='CPU',dtype='int32').realize()
      position=Variable('start_pos',0,511).bind(case['start'])
      result=jit(inputs,position).numpy().ravel()
      if jit.captured is not None and not isinstance(jit.captured,RockchipCapturedJit):jit.captured=RockchipCapturedJit(jit.captured)
      if result.size!=151936 or not np.isfinite(result).all():raise AssertionError('Invalid full-vocabulary output')
      output=directory/(case['name']+'.f32')
      result.tofile(output)
      report['cases'].append({'name':case['name'],'sha256':digest(output),'values':result.size})
      check_sources(hashes)
      save()
      print(json.dumps({'event':'logits','mode':mode,'name':case['name']}),flush=True)
    report['complete']=True
  finally:save()


def run_stock(manifest,directory,binary):
  data=json.loads(manifest.read_text())
  directory.mkdir(parents=True,exist_ok=False)
  report={'complete':False,'manifest_sha256':digest(manifest),'model':str(STOCK_MODEL),'model_sha256':digest(STOCK_MODEL),
          'binary_sha256':digest(binary),'embed_flash':1,'domain':1,'cases':[]}
  target=directory/'report.json'
  def save():target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  log=directory/'runtime.log'
  try:
    command=['flock','/tmp/rk3588_npu_submit.lock','taskset','-c','4-7',sys.executable,'-m','extra.rockchip.sdk_entry',str(binary),str(STOCK_MODEL),
             str(manifest.with_suffix('.tokens')),str(directory)]
    with log.open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    info=[line for line in log.read_text().splitlines() if 'model_dtype:' in line]
    if len(info)!=1 or 'model_dtype: W8A8' not in info[0]:raise AssertionError('RKLLM precision differs')
    report['sdk_model_info']=info[0]
    for case in data['cases']:
      path=directory/(case['name']+'.f32')
      if path.stat().st_size!=151936*4:raise AssertionError('Incomplete RKLLM vocabulary dump')
      report['cases'].append({'name':case['name'],'sha256':digest(path),'values':151936})
    report['complete']=True
  finally:save()


def compare(manifest,reference,stock,candidate,output):
  data=json.loads(manifest.read_text())
  mask=np.arange(151936)!=EOS_ID
  report={'complete':False,'manifest_sha256':digest(manifest),'cases':[],'gate':data['quality_gate']}
  metadata={name:json.loads((path/'report.json').read_text()) for name,path in [('fp16',reference),('rkllm',stock),('int8',candidate)]}
  if any(not m['complete'] or m['manifest_sha256']!=digest(manifest) for m in metadata.values()):
    raise AssertionError('Quality dump provenance differs')
  report['candidate_source_sha256']=metadata['int8']['source_sha256']
  report['candidate_configuration']=metadata['int8']['placement']
  report['checkpoint_sha256']=data['checkpoint_sha256']
  report['stock_model_sha256']=metadata['rkllm']['model_sha256']
  def log_probability(values):
    if values.size!=151936 or not np.isfinite(values[mask]).all():raise AssertionError('Invalid vocabulary output')
    x=values[mask].astype(np.float64)
    maximum=x.max()
    return x-(maximum+np.log(np.exp(x-maximum).sum()))
  indices=np.arange(151936)[mask]
  for case in data['cases']:
    arrays=[np.fromfile(p/(case['name']+'.f32'),dtype=np.float32) for p in (reference,stock,candidate)]
    if not np.isneginf(arrays[1][EOS_ID]):raise AssertionError('Unexpected RKLLM EOS policy')
    probabilities=[log_probability(a) for a in arrays]
    gold=int(np.searchsorted(indices,case['gold']))
    if gold>=len(indices) or indices[gold]!=case['gold']:raise AssertionError('Gold token excluded')
    row={'name':case['name'],'gold':case['gold'],'phase':'prefill' if case['start']==0 else 'decode'}
    for name,values in zip(('fp16','rkllm','int8'),probabilities):
      row[name]={'nll':float(-values[gold]),'argmax':int(indices[values.argmax()]),
                 'kl_from_fp16':float(np.sum(np.exp(probabilities[0])*(probabilities[0]-values)))}
    report['cases'].append(row)
  report['summary']={}
  for name in ('fp16','rkllm','int8'):
    nll=float(np.mean([r[name]['nll'] for r in report['cases']]))
    report['summary'][name]={'mean_nll':nll,'sampled_conditional_perplexity':math.exp(nll),
      'mean_kl_from_fp16':float(np.mean([r[name]['kl_from_fp16'] for r in report['cases']])),
      'argmax_agreement_with_fp16':float(np.mean([r[name]['argmax']==r['fp16']['argmax'] for r in report['cases']]))}
  ratio=math.exp(report['summary']['int8']['mean_nll']-report['summary']['rkllm']['mean_nll'])
  report['perplexity_ratio_by_phase']={phase:math.exp(float(np.mean(
    [r['int8']['nll']-r['rkllm']['nll'] for r in report['cases'] if r['phase']==phase]))) for phase in ('prefill','decode')}
  report.update(complete=True,perplexity_ratio_vs_rkllm=ratio,passed=ratio<=report['gate']['max_perplexity_ratio_vs_rkllm'])
  output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'summary':report['summary'],'ratio':ratio,'passed':report['passed']}),flush=True)
  if not report['passed']:raise AssertionError('Comparable-quality gate failed')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode',choices=['prepare','stock','fp16','int8','compare'])
  parser.add_argument('--manifest',type=Path,required=True)
  parser.add_argument('--directory',type=Path)
  parser.add_argument('--corpus',type=Path)
  parser.add_argument('--contexts',type=int,default=64)
  parser.add_argument('--reference',type=Path)
  parser.add_argument('--stock',type=Path)
  parser.add_argument('--candidate',type=Path)
  parser.add_argument('--rkllm-binary',type=Path)
  parser.add_argument('--cpu-pool',action='store_true')
  parser.add_argument('--cpu-fast-softmax',action='store_true')
  args=parser.parse_args()
  if args.mode=='prepare':prepare(args.corpus,args.manifest,args.contexts)
  elif args.mode=='stock':run_stock(args.manifest,args.directory,args.rkllm_binary)
  elif args.mode=='compare':compare(args.manifest,args.reference,args.stock,args.candidate,args.directory)
  else:run_tinygrad(args.mode,args.manifest,args.directory,args.cpu_pool,args.cpu_fast_softmax)


if __name__=='__main__':main()
