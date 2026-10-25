"""Freeze exact-length image/paragraph prompts for equal-output API/SDK timing.

Use both deterministic records and naturally varied WikiText. Compute pixels
with the API's tensor expression and derive SDK embeddings from stored weights.
This prepares inputs only; no measured output or timing is selected here.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  os.environ['DEV'] = 'CPU'
  from tinygrad import Tensor, dtypes
  from tinygrad.nn.state import safe_load
  from onpu.vlm import decode_image, load_tokenizer, render_messages
  args.output.mkdir(parents=True,exist_ok=False)
  original = args.root/'fix-context-final-vlm-fixtures/length-8184.json'
  payload = json.loads(original.read_text())
  image = next(b for m in payload['messages'] for b in m['content'] if b['type']=='image_url')
  tokenizer = load_tokenizer(args.root/'hf')
  records = tokenizer.encode(''.join(f'Record {i}: The quiet garden has a path beside the wooden fence.\n' for i in range(8192)))
  corpus = args.root/'fix-context-wikitext2-test.txt'
  wiki = tokenizer.encode(corpus.read_text())
  def build(target,filler):
    budget,padding = max(0,target-128),1
    for _ in range(16):
      text = ('Background notes:\n'+tokenizer.decode(filler[:budget])+'\n'+' x'*padding+
              '\nDescribe the supplied image in detail in a paragraph of at least 100 words.')
      messages = [{'role':'user','content':[image,{'type':'text','text':text}]}]
      ids = tokenizer.encode(render_messages(messages,64)[0])
      delta = target-len(ids)
      if delta==0:
        assert ids.count(49190)==64
        return {'model':payload['model'],'messages':messages,'max_tokens':8,'temperature':0,'stream':False},ids
      if padding+delta>=1: padding += delta
      else: budget=max(0,budget+delta-8);padding+=8
    raise AssertionError(f'Could not build {target} positions; got {len(ids)}')
  checkpoint = args.root/'hf/model.safetensors'
  stored = safe_load(checkpoint)['model.text_model.embed_tokens.weight']
  weights = ((stored.bitcast(dtypes.uint16).numpy().astype(np.uint32)<<16).view(np.float32)
             if stored.dtype is dtypes.bfloat16 else stored.numpy().astype(np.float32))
  weights = weights.astype(np.float16).astype(np.float32)
  rgb = decode_image(image['image_url']['url'])
  pixels = ((Tensor(rgb,device='CPU').reshape(512,512,3).permute(2,0,1).unsqueeze(0).float()/255-.5)/.5).contiguous().numpy()
  report = {'complete':False,'context':8192,'output_tokens':8,'checkpoint_sha256':digest(checkpoint),
            'corpus_sha256':digest(corpus),'api_pixel_bytes_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
            'decoded_rgb_sha256':hashlib.sha256(rgb).hexdigest(),'source_payload_sha256':digest(original),
            'source_sha256':digest(__file__),'workloads':{}}
  for label,target,filler in [('warmup128',128,records),('records8184',8184,records),('wiki8184',8184,wiki)]:
    body,ids = build(target,filler)
    directory = args.output/label
    directory.mkdir()
    (directory/'payload.json').write_text(json.dumps(body)+'\n')
    (directory/'input_ids.json').write_text(json.dumps(ids)+'\n')
    weights[np.array(ids)].astype(np.float32).tofile(directory/'input_embeddings.f32')
    np.save(directory/'pixels.npy',pixels)
    report['workloads'][label] = {'positions':len(ids),'payload_sha256':digest(directory/'payload.json'),
                                'input_ids_sha256':digest(directory/'input_ids.json'),
                                'embeddings_sha256':digest(directory/'input_embeddings.f32')}
  report['complete'] = True
  (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)


if __name__=='__main__': main()
