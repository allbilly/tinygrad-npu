"""Read BPE SentencePiece metadata without adding a protobuf/runtime dependency.

Field numbers follow sentencepiece_model.proto. The supported normalization is
the identity, whitespace-preserving form used by TeleChat2's pinned tokenizer.
"""
from __future__ import annotations
import hashlib, math, struct


def fields(data:bytes) -> list[tuple[int,int,int|bytes]]:
  position,result = 0,[]
  def varint():
    nonlocal position
    value = 0
    for shift in range(0,70,7):
      if position>=len(data):raise ValueError('Truncated protobuf varint')
      byte = data[position]
      position+=1
      value|=(byte&127)<<shift
      if not byte&128:
        if value>=2**64:raise ValueError('Oversized protobuf varint')
        return value
    raise ValueError('Oversized protobuf varint')
  while position<len(data):
    tag = varint()
    number,wire = tag>>3,tag&7
    if not number:raise ValueError('Invalid protobuf field number')
    if wire==0:value = varint()
    elif wire in (1,2,5):
      size = varint() if wire==2 else 8 if wire==1 else 4
      if size>len(data)-position:raise ValueError('Truncated protobuf field')
      value = data[position:position+size]
      position+=size
    else:raise ValueError('Unsupported protobuf wire type')
    result.append((number,wire,value))
  return result


def _message(data:bytes) -> dict[int,int|bytes]:
  return {number:value for number,_,value in fields(data)}


def tokenizer_metadata(data:bytes, config:dict) -> dict:
  root = fields(data)
  messages = {number:value for number,wire,value in root if wire==2 and number in (2,3)}
  trainer,normalizer = _message(messages[2]),_message(messages[3])
  if trainer.get(3,1)!=2:raise ValueError('Only SentencePiece BPE is supported')
  if (normalizer.get(1,b'').decode()!='identity' or normalizer.get(2,b'') or normalizer.get(4,1) or not normalizer.get(5,1)):
    raise ValueError('Unsupported SentencePiece normalization')
  if not trainer.get(35,0):raise ValueError('SentencePiece byte fallback is required')
  tokens,scores,types = [],[],[]
  for number,wire,value in root:
    if number!=1:continue
    if wire!=2:raise ValueError('Invalid SentencePiece piece')
    piece = _message(value)
    token,score,kind = piece[1].decode(),struct.unpack('<f',piece.get(2,b'\0'*4))[0],piece.get(3,1)
    if not math.isfinite(score) or kind not in range(1,7):raise ValueError('Invalid SentencePiece piece data')
    tokens.append(token)
    scores.append(score)
    types.append(kind)
  if not tokens or len(set(tokens))!=len(tokens):raise ValueError('Invalid SentencePiece vocabulary')
  by_token = {token:index for index,token in enumerate(tokens)}
  def token_id(field,default):
    value = trainer.get(field,default)
    return value-2**64 if value>=2**63 else value
  unk,bos,eos,pad = (token_id(field,default) for field,default in ((40,0),(41,1),(42,2),(43,-1)))
  if not 0<=unk<len(tokens) or types[unk]!=2:raise ValueError('Invalid SentencePiece unknown piece')
  registered = {value for value in (unk,bos,eos,pad) if value>=0}
  for key,item in config.get('added_tokens_decoder',{}).items():
    index = int(key)
    if not 0<=index<len(tokens) or tokens[index]!=item['content'] or not item.get('special'):
      raise ValueError('Added tokens must match embedded special pieces')
    registered.add(index)
  for token in config.get('additional_special_tokens',[]):
    if token not in by_token:raise ValueError('Added special token has no model embedding')
    registered.add(by_token[token])
  result = {'tokenizer.ggml.model':'llama','tokenizer.ggml.tokens':tokens,'tokenizer.ggml.scores':scores,
            'tokenizer.ggml.token_type':types,'tokenizer.ggml.add_space_prefix':bool(normalizer.get(3,1)),
            'tokenizer.ggml.add_bos_token':bool(config.get('add_bos_token',False)),
            'tokenizer.ggml.bos_token_id':bos,'tokenizer.ggml.eos_token_id':eos,
            'tokenizer.tinygrad.special_token_ids':sorted(registered),
            'tokenizer.tinygrad.source_sha256':hashlib.sha256(data).hexdigest(),
            'tokenizer.tinygrad.unknown_piece':tokens[unk]}
  if pad>=0:result['tokenizer.ggml.padding_token_id'] = pad
  return result
