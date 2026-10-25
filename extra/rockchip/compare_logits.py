"""Compare every FP16 Qwen3 logit, including one stateful decode step, with RKLLM."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np

EOS_ID = 151645  # Checked against GGUF metadata after loading.


def digest(path):
  with path.open('rb') as source: return hashlib.file_digest(source,'sha256').hexdigest()


def metrics(actual,expected):
  finite = np.isfinite(expected)
  excluded = np.where(~finite)[0].tolist()
  if (actual.shape!=expected.shape or not np.isfinite(actual).all()
      or excluded not in ([],[EOS_ID]) or excluded and not np.isneginf(expected[EOS_ID])):
    raise AssertionError('Unexpected shape or nonfinite logits')
  a,b = actual[finite].astype(np.float64),expected[finite].astype(np.float64)
  error = a-b
  indices = np.where(finite)[0]
  row = {'count':a.size,'total_count':actual.size,'stock_masked_ids':excluded,
         'max_abs':float(np.abs(error).max()),'rmse':float(np.sqrt(np.mean(error**2))),
         'cosine':float(a@b/np.sqrt((a@a)*(b@b))),'argmax':int(indices[a.argmax()]),'reference_argmax':int(indices[b.argmax()])}
  row['within_limits'] = row['max_abs']<=.25 and row['rmse']<=.02 and row['cosine']>=.9999 and row['argmax']==row['reference_argmax']
  return row


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report-dir',type=Path,required=True)
  parser.add_argument('--stock-dumps',type=Path,help='Reuse stock dumps from an earlier run with the identical stock binary/model hashes')
  parser.add_argument('--fused-qkv',action='store_true')
  parser.add_argument('--cpu-norm',action='store_true')
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  matched = Path('/home/orangepi/qwen3-bench/matched')
  checkpoint = matched/'roofline/gpu/Qwen3-0.6B-matched-f16.gguf'
  binary,stock_model = matched/'rkllm_fp16_bench',matched/'Qwen3-0.6B-fp16-3core.rkllm'
  args.report_dir.mkdir(parents=True,exist_ok=False)
  prompt128 = [int(x) for x in (matched/'roofline/final-stock128/input.tokens').read_text().split()]
  prompt256 = [int(x) for x in (matched/'roofline/final-stock256/input.tokens').read_text().split()]
  cases = [('prefill128',prompt128),('decode129',prompt128+[4379]),('prefill256',prompt256)]
  sources = [Path(__file__),*(root/name for name in ['tinygrad/runtime/ops_rockchip.py','tinygrad/llm/model.py',
    'tinygrad/llm/kernels/rockchip.py','tinygrad/llm/kernels/rockchip_cpu.py','tinygrad/llm/kernels/rockchip_replay.py',
    'tinygrad/llm/kernels/rockchip_qkv.py'])]
  hashes = {str(path.relative_to(root)):digest(path) for path in sources}
  report = {'complete':False,'limits':{'max_abs':.25,'rmse':.02,'cosine_min':.9999,'same_argmax':True},
            'fused_qkv':args.fused_qkv,'cpu_norm':args.cpu_norm,
            'source_sha256':hashes,'checkpoint_sha256':digest(checkpoint),'stock_binary_sha256':digest(binary),
            'stock_model_sha256':digest(stock_model),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cases':{}}
  target = args.report_dir/'comparison.json'
  def save(): target.write_text(json.dumps(report,indent=2)+'\n')
  save()
  if args.stock_dumps:
    prior = json.loads((args.stock_dumps/'comparison.json').read_text())
    if any(prior[key]!=report[key] for key in ['stock_binary_sha256','stock_model_sha256']):
      raise AssertionError('Reused stock dump provenance differs')
  expected = {}
  env = {**os.environ,'LOGITS_ONLY':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','RKLLM_EMBED_FLASH':'0','NPU_DOMAIN_ID':'1'}
  for name,tokens in cases:
    input_path = args.report_dir/(name+'.tokens')
    input_path.write_text(' '.join(map(str,tokens))+'\n')
    dump = args.report_dir/(name+'-stock.f32')
    command = ['flock','/tmp/rk3588_npu_submit.lock','taskset','-c','4-7',str(binary),str(stock_model),str(input_path),'32','1',str(dump)]
    prior_dump = args.stock_dumps/(name+'-stock.f32') if args.stock_dumps else None
    if prior_dump is not None and prior_dump.exists():
      prior_tokens = [int(x) for x in (args.stock_dumps/(name+'.tokens')).read_text().split()]
      if prior_tokens!=tokens: raise AssertionError('Reused stock input IDs differ')
      dump.write_bytes(prior_dump.read_bytes())
    else:
      with (args.report_dir/(name+'-stock.log')).open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    expected[name] = np.fromfile(dump,dtype=np.float32)
    if expected[name].size!=151936: raise AssertionError('Incomplete stock logits')
    metrics(np.ones(151936,dtype=np.float32),expected[name])  # Validate masks before claiming a complete dump.
    if name=='prefill128' and int(expected[name].argmax())!=4379: raise AssertionError('Decode input is not the stock greedy token')
    report['cases'][name] = {'input_ids':tokens,'stock_dump_sha256':digest(dump)}
    save()
    print(json.dumps({'event':'stock_logits','case':name,'count':expected[name].size}),flush=True)
  os.environ['DEV'] = 'CPU'
  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.kernels.rockchip import RockchipRMSNorm, configure
  model,metadata = Transformer.from_gguf(checkpoint,max_context=512,realize=True)
  if metadata['tokenizer.ggml.eos_token_id']!=EOS_ID: raise AssertionError('Unexpected EOS token metadata')
  report['stock_mask_explanation'] = 'RKLLM ignore_eos_token=true returns -inf at the GGUF EOS ID; all other vocabulary logits are compared'
  report['placement'] = configure(model,prefill='on',decode='on',tile_columns=24576,fuse_projections=True,
                                  cpu_attention=True,native_replay=False,fused_ffn=True,split_down=True,
                                  fused_qkv=args.fused_qkv,cpu_norm=args.cpu_norm)
  split = [(block.fused_ffn,block.fused_ffn.split_weight) for block in model.blk]
  prepared = [(block,getattr(block,'prepared_qkv',None)) for block in model.blk]
  norms = [(obj,name,norm) for obj,name in [*((b,n) for b in model.blk for n in ('attn_norm','ffn_norm')),(model,'output_norm')]
           if isinstance(norm:=getattr(obj,name),RockchipRMSNorm)]
  baseline = {}
  try:
    for mode in ['full_k','split_k']:
      for ffn,weight in split: ffn.split_weight = weight if mode=='split_k' else None
      for block,kernel in prepared: block.prepared_qkv = kernel if mode=='split_k' else None
      for obj,name,norm in norms: setattr(obj,name,norm if mode=='split_k' else norm.source)
      previous = []
      for name,tokens in cases:
        start = len(previous) if previous and tokens[:len(previous)]==previous else 0
        result = model.logits(Tensor([tokens[start:]],device='CPU',dtype='int32'),start).numpy().flatten()
        if result.size!=151936: raise AssertionError('Incomplete tinygrad logits')
        dump = args.report_dir/(name+'-'+mode+'.f32')
        result.tofile(dump)
        row = metrics(result,expected[name])
        row.update(input_rows=len(tokens)-start,start_pos=start,dump_sha256=digest(dump))
        if mode=='full_k': baseline[name] = result
        else: row['versus_full_k'] = metrics(result,baseline[name])
        report['cases'][name][mode] = row
        previous = tokens
        save()
        print(json.dumps({'event':'tinygrad_logits','case':name,'mode':mode,**row}),flush=True)
    if any(digest(root/name)!=value for name,value in hashes.items()): raise AssertionError('Measured source changed')
    report['complete'] = True
    report['passed'] = all(report['cases'][name][mode]['within_limits'] for name,_ in cases for mode in ['full_k','split_k']) \
      and all(report['cases'][name]['split_k']['versus_full_k']['within_limits'] for name,_ in cases)
    save()
    if not report['passed']: raise AssertionError('Full-vocabulary logit limits failed; inspect comparison.json')
  finally:
    for ffn,weight in split: ffn.split_weight = weight
    for block,kernel in prepared: block.prepared_qkv = kernel
    for obj,name,norm in norms: setattr(obj,name,norm)


if __name__=='__main__': main()
