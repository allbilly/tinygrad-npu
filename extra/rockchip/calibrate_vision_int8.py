"""Calibrate SmoothQuant channel scales from the demo image only.

For each projection, X/s and W*s preserve the real-valued linear operation.
The held-out question/image fixtures are deliberately not read by this program.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from transformers import Idefics3ForConditionalGeneration

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--pixels',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  model = Idefics3ForConditionalGeneration.from_pretrained(args.checkpoint,torch_dtype=torch.float32,
                                                          attn_implementation='eager',local_files_only=True).eval()
  vision = model.model.vision_model
  inputs = {}
  modules = dict(vision.named_modules())
  handles = []
  for name,module in modules.items():
    if not isinstance(module,torch.nn.Linear): continue
    def capture(module,arguments,name=name):
      value = arguments[0].detach().reshape(-1,arguments[0].shape[-1]).abs().amax(0)
      inputs[name] = torch.maximum(inputs[name],value) if name in inputs else value
    handles.append(module.register_forward_pre_hook(capture))
  with torch.inference_mode(): vision(torch.from_numpy(np.load(args.pixels)))
  for handle in handles: handle.remove()
  scales = {}
  alpha = .5
  for name,amax in inputs.items():
    if name.endswith(('.q_proj','.k_proj','.v_proj')):
      prefix = name.rsplit('.',1)[0]
      output_name = prefix+'.qkv'
      weight = torch.cat([modules[prefix+'.'+member].weight for member in ('q_proj','k_proj','v_proj')],dim=0)
    else: output_name,weight = name,modules[name].weight
    wmax = weight.detach().abs().amax(0)
    scale = (amax.clamp(min=1e-5).pow(alpha)/wmax.clamp(min=1e-5).pow(1-alpha)).clamp(1e-4,1e4)
    scale = torch.where((amax==0)|(wmax==0),torch.ones_like(scale),scale)
    output_name = output_name.replace('encoder.layers.','layers.')
    if output_name in scales:
      np.testing.assert_array_equal(scales[output_name],scale.numpy())
    scales[output_name] = scale.numpy().astype(np.float32)
  np.savez(args.output,**scales)
  metadata = {'checkpoint_sha256':hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
              'pixels_sha256':hashlib.sha256(args.pixels.read_bytes()).hexdigest(),'alpha':alpha,'projections':len(scales),
              'scale_min':min(float(s.min()) for s in scales.values()),'scale_max':max(float(s.max()) for s in scales.values()),
              'calibration':'demo image only; FP32 encoder projection inputs; no held-out fixtures'}
  args.output.with_suffix('.json').write_text(json.dumps({**metadata,'scales':{k:v.tolist() for k,v in scales.items()}},indent=2)+'\n')
  print(json.dumps(metadata),flush=True)

if __name__ == '__main__': main()
