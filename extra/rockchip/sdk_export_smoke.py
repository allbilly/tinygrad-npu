"""Small real SDK export smoke test; synthetic weights are not a model benchmark."""
import argparse, json, shutil
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--tokenizer',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True,exist_ok=False)
  import torch
  from transformers import LlamaConfig, LlamaForCausalLM
  from extra.rockchip.rkllm_reference import export_reference, validate_reference
  torch.set_num_threads(1)
  torch.manual_seed(0)
  config = LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                       vocab_size=49280,max_position_embeddings=512,bos_token_id=1,eos_token_id=49279,tie_word_embeddings=True)
  model = LlamaForCausalLM(config)
  directory = args.output/'hf'
  model.save_pretrained(directory)
  del model
  for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
    if (args.tokenizer/name).exists():shutil.copyfile(args.tokenizer/name,directory/name)
  dataset = args.output/'calibration.json'
  dataset.write_text(json.dumps([{'input':'Hello world.','target':'Hello.'}]))
  artifact = args.output/'synthetic-w8a8-opt0-ctx512.rkllm'
  export_reference(directory,dataset,artifact)
  validate_reference(artifact,context=512)
  print(json.dumps({'complete':True,'scope':'synthetic SDK export only','artifact':str(artifact)}),flush=True)


if __name__=='__main__':main()
