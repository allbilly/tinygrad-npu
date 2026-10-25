"""Compare complete GPT-2 vocabulary logits and greedy continuations with CPU."""
import argparse, hashlib, json, os
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--cpu-operands',choices=['stock','fp16'],default='stock',help='Choose ordinary CPU arithmetic or matched FP16 matmul operands')
  args = parser.parse_args()
  os.environ.update(DEV='CPU',HALF='1',MAX_CONTEXT='1024',JIT_BATCH_SIZE='0')
  import numpy as np
  from tinygrad import Device, Tensor, Variable, nn
  from unittest.mock import patch
  from examples.gpt2 import GPT2, VOCAB_SIZE
  from tinygrad.llm.kernels.rockchip_gpt2 import configure_gpt2
  from tinygrad.llm.kernels.rockchip import pack_weight, packed_matmul

  model = GPT2.build('gpt2')
  transformer = model.model
  text = 'The quick brown fox jumps over the lazy dog. The capital of France is Paris. A computer processes information using arithmetic. '
  prompts = [model.tokenizer.encode(text*20)[:length] for length in (8,32,128)]
  prompts += [model.tokenizer.encode(t) for t in ['Hello, my name is','Write a Python function that adds two numbers.']]
  report = {'complete':False,'context':1024,'tokens':32,'thresholds':{'max_abs':.1,'rmse':.025,'kl_divergence':1e-4},
            'cpu_operands':args.cpu_operands,'cases':[],'matmuls':[],'cpu_references':[],'failures':[]}
  root = Path(__file__).resolve().parents[2]
  report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__),root/'examples/gpt2.py',*(root/'tinygrad').rglob('*rockchip*.py'),*(root/'tinygrad/llm/kernels').glob('rockchip*.c')]}
  args.report.parent.mkdir(parents=True,exist_ok=True)
  def save(): args.report.write_text(json.dumps(report,indent=2)+'\n')
  def reset():
    for block in transformer.h:
      if hasattr(block.attn,'cache_kv'): block.attn.cache_kv.assign(Tensor.zeros_like(block.attn.cache_kv)).realize()
  def request(prompt):
    reset()
    inp = Tensor([prompt],device='CPU').realize()
    position = Variable('start_pos',0,1023).bind(0)
    logits = transformer.logits(inp,position).numpy().astype(np.float64).flatten()
    first = int(logits.argmax())
    generated = [first]
    decoded = []
    for i in range(31):
      token = Variable('tokens',0,VOCAB_SIZE-1).bind(generated[-1])
      position = Variable('start_pos',1,1023).bind(len(prompt)+i)
      if i in (0,15,30): decoded.append(transformer.logits(token,position).numpy().astype(np.float64).flatten())
      generated.extend(transformer(token,position,0.0).tolist())
    return logits,decoded,generated
  try:
    references = []
    original = nn.Linear.__call__
    def cpu_linear(layer, x):
      if args.cpu_operands=='stock': return original(layer,x)
      out = x.half().float().linear(layer.weight.half().float().T).cast(x.dtype)
      return out if layer.bias is None else out+layer.bias
    with patch.object(nn.Linear,'__call__',cpu_linear):
      for index,prompt in enumerate(prompts):
        references.append(request(prompt))
        report['cpu_references'].append({'input_ids':prompt,'generated_ids':references[-1][2]})
        save()
        print(json.dumps({'event':'cpu_reference','case':index,'input_tokens':len(prompt)}),flush=True)
    configure_gpt2(transformer)
    for index,(prompt,(reference,decoded_reference,expected)) in enumerate(zip(prompts,references)):
      actual,decoded,generated = request(prompt)
      comparisons = []
      for label,a,b in [('prefill',actual,reference),*((f'decode-{i}',a,b) for i,(a,b) in enumerate(zip(decoded,decoded_reference)))]:
        error = a-b
        p,q = np.exp(b-b.max()),np.exp(a-a.max())
        p,q = p/p.sum(),q/q.sum()
        comparisons.append({'phase':label,'values':len(a),'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
                            'max_abs':float(np.abs(error).max()),'rmse':float(np.sqrt(np.mean(error**2))),
                            'mean_offset':float(error.mean()),'centered_rmse':float(np.std(error)),
                            'kl_divergence':float(np.sum(p*np.log(p/q))),'argmax_matches':bool(a.argmax()==b.argmax())})
      row = {'input_ids':prompt,'generated_ids':generated,'matches_cpu':generated==expected,'logits':comparisons}
      report['cases'].append(row)
      save()
      print(json.dumps({'event':'candidate','case':index,**row}),flush=True)
      if generated!=expected: report['failures'].append(f'Greedy continuation differs in case {index}')
      for check in comparisons:
        if not check['finite'] or not check['argmax_matches'] or any(check[k]>v for k,v in report['thresholds'].items()):
          report['failures'].append(f'Full-vocabulary logit comparison failed in case {index}: {check}')
    # Complete-output coverage of every selected GPT-2 weight geometry and the
    # bounded prefill path. New activations rebind the same prepared JIT weights.
    rng = np.random.default_rng(1819)
    from tinygrad import TinyJit
    for rows,depth,columns in [(1,768,768),(1,768,2304),(1,768,3072),(1,3072,768),(1,768,24576),
                               (64,768,2304),(64,768,3072),(21,3072,768)]:
      weights = rng.normal(0,.1,(columns,depth)).astype(np.float16)
      packed = pack_weight(Tensor(weights,device='CPU')).realize()
      fn = TinyJit(lambda x:packed_matmul(x,packed,columns).realize())
      maximum = 0.
      for _ in range(4):
        data = rng.normal(0,.1,(rows,depth)).astype(np.float16)
        actual = fn(Tensor(data,device='CPU').realize()).numpy()
        expected_matrix = data.astype(np.float32)@weights.astype(np.float32).T
        np.testing.assert_allclose(actual,expected_matrix,atol=2e-4,rtol=3e-4)
        maximum = max(maximum,float(np.abs(actual-expected_matrix).max()))
      report['matmuls'].append({'rows':rows,'depth':depth,'columns':columns,'values_checked':rows*columns*4,'max_abs':maximum})
      save()
    report.update(npu_submits=Device['ROCKCHIP'].submit_count,npu_tasks=Device['ROCKCHIP'].task_count,
                  probability_checks_passed=all(c['finite'] and c['argmax_matches'] and c['kl_divergence']<=1e-4
                                                for row in report['cases'] for c in row['logits']),
                  greedy_checks_passed=all(row['matches_cpu'] for row in report['cases']))
    if report['failures']: raise AssertionError(f"{len(report['failures'])} full-vocabulary checks failed; see report")
    report['complete'] = True
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    raise
  finally: save()


if __name__=='__main__': main()
