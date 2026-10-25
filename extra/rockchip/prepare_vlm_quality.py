"""Create disjoint image/question fixtures without downloading extra datasets."""
import argparse, json, subprocess, sys
from pathlib import Path

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  args = parser.parse_args()
  cases = json.loads((args.root/'quality-cases.json').read_text())
  for index,case in enumerate(cases):
    output = args.root/'quality'/case['id']
    if (output/'teacher_logits.npy').exists(): continue
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.log').open('w') as log:
      subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_vlm_reference.py')),
        '--checkpoint',str(args.root/'hf'),'--image',case['image'],'--question',case['question'],'--tokens','16','--reuse-vision',
        '--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'case':case['id'],'index':index,'complete':True}),flush=True)

if __name__ == '__main__': main()
