"""Qualify and time the staged encoder before integrating it into generation."""
import argparse, faulthandler, hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

def main():
  faulthandler.register(signal.SIGUSR1)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--runs',type=int,default=6)
  parser.add_argument('--layers',type=int,default=12)
  parser.add_argument('--attention-device',choices=['CL','CPU','ROCKCHIP'],default='CL')
  parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--profile',action='store_true')
  parser.add_argument('--calibration',type=Path)
  parser.add_argument('--npu-connector',action='store_true')
  parser.add_argument('--fused-mlp',action='store_true')
  parser.add_argument('--batch-prefill',action='store_true')
  args = parser.parse_args()
  os.environ['DEV'] = 'CPU'
  os.environ['ROCKCHIP_BINDING_CACHE'] = '4096'
  from tinygrad import Tensor, Device
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision, load_input_scales
  model = SmolVLM.from_hf(args.root/'hf',512)
  model.vision.layers = model.vision.layers[:args.layers]
  scales = None
  if args.calibration:
    with (args.root/'hf/model.safetensors').open('rb') as file: checkpoint_hash = hashlib.file_digest(file,'sha256').hexdigest()
    scales = load_input_scales(args.calibration,checkpoint_hash)
  placement = configure(model.vision,prefill='on',decode='off',precision=args.precision,input_scales=scales,batch_prefill=args.batch_prefill)
  encoder = StagedVision(model,args.attention_device,profile=args.profile,npu_connector=args.npu_connector,fused_mlp=args.fused_mlp)
  print(json.dumps({'event':'ready','placement':placement}),flush=True)
  pixels = Tensor(np.load(args.root/'demo/pixels.npy')).realize()
  reference = np.load(args.root/'demo/image_embeddings.npy')
  previous = None
  for index in range(args.runs):
    submits = Device['ROCKCHIP'].submit_count
    start = time.perf_counter()
    output = encoder(pixels).numpy()
    ms = (time.perf_counter()-start)*1000
    if previous is not None: np.testing.assert_allclose(output,previous,atol=1e-4,rtol=1e-4)
    previous = output.copy()
    row = {'event':'iteration','index':index,'ms':ms,'submits':Device['ROCKCHIP'].submit_count-submits,
           'finite':bool(np.isfinite(output).all()),'nrmse':float(np.linalg.norm(output-reference)/np.linalg.norm(reference)),
           'native':encoder.statistics(),'stages_ms':encoder.timings}
    print(json.dumps(row),flush=True)

if __name__ == '__main__': main()
