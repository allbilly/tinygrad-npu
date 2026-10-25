"""Convert verified TeleChat2 HF shards to a streaming FP16 GGUF for tinygrad.

Uses the existing PyTorch installation only for weights-only, memory-mapped
loading. No downloaded Python model code runs. Extended dynamic NTK beyond the
training context is not implemented; the GGUF exposes the supported short range.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from extra.rockchip.sentencepiece_metadata import tokenizer_metadata


@dataclass(frozen=True)
class Mapping:
  name:str
  source:str
  shape:tuple[int,...]
  source_shape:tuple[int,...]
  kv_component:int = -1
  head_dim:int = 0


def mappings(config:dict, keys:set[str]) -> list[Mapping]:
  if config['model_type']!='telechat' or config.get('apply_residual_connection_post_layernorm',False):
    raise ValueError('Expected the TeleChat2 pre-normalization decoder')
  dim,hidden,vocab,layers,heads = (config[k] for k in ('hidden_size','ffn_hidden_size','vocab_size','n_layer','n_head'))
  kv_heads = config.get('num_key_value_heads') or heads
  if dim%heads or heads%kv_heads:raise ValueError('Invalid head dimensions')
  head = dim//heads
  result = [Mapping('token_embd.weight','transformer.word_embeddings.weight',(vocab,dim),(vocab,dim)),
            Mapping('output_norm.weight','transformer.ln_f.weight',(dim,),(dim,)),
            Mapping('output.weight','lm_head.weight',(vocab,dim),(vocab,dim))]
  for index in range(layers):
    prefix,target = f'transformer.h.{index}.',f'blk.{index}.'
    for dest,src,shape in [('attn_norm.weight','input_layernorm.weight',(dim,)),
                           ('attn_q.weight','self_attention.query.weight',(dim,dim)),
                           ('attn_output.weight','self_attention.dense.weight',(dim,dim)),
                           ('attn_output.bias','self_attention.dense.bias',(dim,)),
                           ('ffn_norm.weight','post_attention_layernorm.weight',(dim,)),
                           ('ffn_gate.weight','mlp.gate_proj.weight',(hidden,dim)),
                           ('ffn_up.weight','mlp.up_proj.weight',(hidden,dim)),
                           ('ffn_down.weight','mlp.down_proj.weight',(dim,hidden)),
                           ('ffn_down.bias','mlp.down_proj.bias',(dim,))]:
      result.append(Mapping(target+dest,prefix+src,shape,shape))
    for part,component in (('k',0),('v',1)):
      result.append(Mapping(target+'attn_'+part+'.weight',prefix+'self_attention.key_value.weight',
                            (kv_heads*head,dim),(2*kv_heads*head,dim),component,head))
  expected = {item.source for item in result}
  if expected!=keys:raise ValueError(f'Unsupported source tensors: missing={sorted(expected-keys)}, extra={sorted(keys-expected)}')
  return result


def source_rows(mapping:Mapping, start:int, count:int) -> list[int]:
  if mapping.kv_component<0:return list(range(start,start+count))
  return [row//mapping.head_dim*(2*mapping.head_dim)+row%mapping.head_dim+mapping.kv_component*mapping.head_dim
          for row in range(start,start+count)]


def metadata(config:dict, tokenizer:dict, source:dict, template:str|None=None) -> dict:
  if any(not key.startswith('tokenizer.') for key in tokenizer):raise ValueError('Expected tokenizer metadata only')
  if len(tokenizer['tokenizer.ggml.tokens'])!=config['vocab_size']:raise ValueError('Tokenizer and embedding vocabulary differ')
  supported = min(config['training_seqlen'],config['base_seqlen'])
  dim,heads = config['hidden_size'],config['n_head']
  result = {'general.architecture':'telechat2','general.type':'model','general.name':'TeleChat2-'+source['size'],
            'general.file_type':1,'general.alignment':32,'general.source.repository':source['repo'],
            'general.source.revision':source['revision'],'general.source.context_length':config['seq_length'],
            'telechat2.context_length':supported,'telechat2.embedding_length':dim,'telechat2.block_count':config['n_layer'],
            'telechat2.feed_forward_length':config['ffn_hidden_size'],'telechat2.attention.head_count':heads,
            'telechat2.attention.head_count_kv':config.get('num_key_value_heads') or heads,
            'telechat2.attention.key_length':dim//heads,'telechat2.attention.value_length':dim//heads,
            'telechat2.attention.layer_norm_rms_epsilon':float(config['layer_norm_epsilon']),
            'telechat2.rope.dimension_count':dim//heads,'telechat2.rope.freq_base':10000.,**tokenizer}
  if template:
    result['tokenizer.chat_template'] = template.replace("message.role == 'bot'","message.role in ['bot', 'assistant']")
  return result


def _string(value:str) -> bytes:
  data = value.encode()
  return struct.pack('<Q',len(data))+data


def _scalar(value) -> tuple[int,bytes]:
  if type(value) is bool:return 7,struct.pack('<?',value)
  if type(value) is int:
    kind,fmt = (4,'I') if 0<=value<2**32 else (5,'i') if -2**31<=value<0 else (10,'Q') if value>=0 else (11,'q')
    return kind,struct.pack('<'+fmt,value)
  if type(value) is float:return 6,struct.pack('<f',value)
  if type(value) is str:return 8,_string(value)
  raise ValueError('Unsupported GGUF metadata value')


def _value(value) -> bytes:
  if isinstance(value,list):
    if not value:raise ValueError('Empty metadata arrays need an explicit element type')
    encoded = [_scalar(v) for v in value]
    kind = encoded[0][0]
    if any(k!=kind for k,_ in encoded):raise ValueError('Metadata array types must agree')
    return struct.pack('<IIQ',9,kind,len(encoded))+b''.join(data for _,data in encoded)
  kind,data = _scalar(value)
  return struct.pack('<I',kind)+data


def write_f16(path:Path, kv:dict, tensors:list[Mapping], payloads:Callable[[Mapping],Iterable[bytes]]) -> None:
  """Write one new GGUF; caller provides bounded row chunks in little-endian FP16."""
  if len({item.name for item in tensors})!=len(tensors):raise ValueError('Duplicate tensor names')
  infos,offset = [],0
  for item in tensors:
    if not item.shape or any(type(dim) is not int or dim<=0 for dim in item.shape):raise ValueError('Invalid tensor shape')
    offset = (offset+31)//32*32
    infos.append((item,offset))
    offset+=math.prod(item.shape)*2
  header = struct.pack('<4sIQQ',b'GGUF',3,len(tensors),len(kv))
  header+=b''.join(_string(key)+_value(value) for key,value in kv.items())
  for item,offset in infos:
    header+=_string(item.name)+struct.pack('<I',len(item.shape))+struct.pack('<'+'Q'*len(item.shape),*reversed(item.shape))
    header+=struct.pack('<IQ',1,offset)
  base = (len(header)+31)//32*32
  created = False
  try:
    with path.open('xb') as output:
      created = True
      output.write(header+b'\0'*(base-len(header)))
      for item,offset in infos:
        output.write(b'\0'*(base+offset-output.tell()))
        remaining = math.prod(item.shape)*2
        for data in payloads(item):
          if len(data)>remaining or len(data)%2:raise ValueError('FP16 payload exceeds tensor shape')
          output.write(data)
          remaining-=len(data)
        if remaining:raise ValueError('Incomplete FP16 tensor payload')
  except BaseException:
    if created:path.unlink()
    raise


def file_digest(path:Path) -> str:
  with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source',type=Path,required=True)
  parser.add_argument('--tokenizer-kv',type=Path,help='Optional previously exported tokenizer metadata')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--chunk-rows',type=int,default=256)
  args = parser.parse_args()
  if args.chunk_rows<1:parser.error('Need positive chunk rows')
  if args.output.exists():raise FileExistsError('Refusing to replace an existing checkpoint')
  manifest = json.loads((args.source/'source-manifest.json').read_text())
  if not manifest.get('complete'):raise ValueError('Source snapshot is incomplete')
  for item in [*manifest['files'],*manifest['weights']]:
    name = item.get('filename')
    if not name or Path(name).name!=name or not item.get('complete'):raise ValueError('Invalid source manifest entry')
    if file_digest(args.source/name)!=item['sha256']:raise ValueError('Source file checksum differs: '+name)
  config = json.loads((args.source/'config.json').read_text())
  index = json.loads((args.source/'pytorch_model.bin.index.json').read_text())['weight_map']
  if set(index.values())!={item['filename'] for item in manifest['weights']}:raise ValueError('Index and verified weight shards differ')
  verified_metadata = {item['filename'] for item in manifest['files']}
  if not {'config.json','pytorch_model.bin.index.json','tokenizer.model','tokenizer_config.json'}<=verified_metadata:
    raise ValueError('Required source metadata is not verified')
  tokenizer_config = json.loads((args.source/'tokenizer_config.json').read_text())
  tokenizer = (json.loads(args.tokenizer_kv.read_text()) if args.tokenizer_kv else
               tokenizer_metadata((args.source/'tokenizer.model').read_bytes(),tokenizer_config))
  plan = mappings(config,set(index))
  template = tokenizer_config.get('chat_template')
  kv = metadata(config,tokenizer,manifest,template)
  import torch
  cache = {}
  def payloads(item):
    filename = index[item.source]
    if filename not in cache:cache[filename] = torch.load(args.source/filename,map_location='cpu',weights_only=True,mmap=True)
    tensor = cache[filename][item.source]
    if tuple(tensor.shape)!=item.source_shape:raise ValueError('Unexpected source shape: '+item.source)
    for start in range(0,item.shape[0],args.chunk_rows):
      count = min(args.chunk_rows,item.shape[0]-start)
      part = tensor[start:start+count] if item.kv_component<0 else tensor[source_rows(item,start,count)]
      yield part.detach().to(torch.float16).contiguous().numpy().astype('<f2',copy=False).tobytes()
  args.output.parent.mkdir(parents=True,exist_ok=True)
  temporary = args.output.with_name('.'+args.output.name+'.'+uuid.uuid4().hex+'.part')
  try:
    write_f16(temporary,kv,plan,payloads)
    digest = file_digest(temporary)
    os.link(temporary,args.output)  # Atomic publication without replacing another file.
  finally:
    if temporary.exists():temporary.unlink()
  result = {'complete':True,'checkpoint':str(args.output.resolve()),'sha256':digest,'bytes':args.output.stat().st_size,
            'source_repo':manifest['repo'],'source_revision':manifest['revision'],'dtype':'FP16','architecture':'telechat2',
            'source_context':config['seq_length'],'supported_context':kv['telechat2.context_length'],
            'tokenizer_metadata_sha256':hashlib.sha256(json.dumps(tokenizer,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'converter_sha256':file_digest(Path(__file__)),
            'torch_version':torch.__version__,'chunk_rows':args.chunk_rows,'tensor_count':len(plan)}
  args.output.with_suffix('.conversion.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result))


if __name__=='__main__':main()
