import struct, unittest
from extra.rockchip.sentencepiece_metadata import fields, tokenizer_metadata
from tinygrad.llm.cli import SimpleTokenizer


def varint(value):
  result = bytearray()
  while value>=128:
    result.append((value&127)|128)
    value>>=7
  return bytes(result)+bytes([value])


def integer(number,value):return varint(number<<3)+varint(value%(1<<64))
def message(number,value):return varint(number<<3|2)+varint(len(value))+value


class TestSentencePieceMetadata(unittest.TestCase):
  def model(self, *, algorithm=2, normalization='identity', remove_spaces=False, byte_fallback=True):
    # A ModelProto fixture, encoded independently using sentencepiece_model.proto's field numbers.
    pieces = [('<_unk>',2),('<s>',3),('</s>',3)]+[(f'<0x{i:02X}>',6) for i in range(256)]
    pieces += [('▁',1),('a',1),('7',4),('\n',4),('<reserve>',3)]
    data = b''.join(message(1,message(1,token.encode())+varint(2<<3|5)+struct.pack('<f',-float(i))+integer(3,kind))
                    for i,(token,kind) in enumerate(pieces))
    trainer = integer(3,algorithm)+integer(35,byte_fallback)+integer(40,0)+integer(41,1)+integer(42,2)+integer(43,-1)
    normalizer = message(1,normalization.encode())+integer(3,1)+integer(4,remove_spaces)+integer(5,1)
    return data+message(2,trainer)+message(3,normalizer),pieces

  def test_trained_unknown_and_user_symbols_round_trip_without_extra_boundaries(self):
    data,pieces = self.model()
    config = {'unk_token':'<unk>','add_bos_token':False,'additional_special_tokens':['<s>','</s>']}
    kv = tokenizer_metadata(data,config)
    self.assertEqual(kv['tokenizer.ggml.tokens'],[token for token,_ in pieces])
    self.assertEqual(kv['tokenizer.tinygrad.unknown_piece'],'<_unk>')
    self.assertEqual(kv['tokenizer.tinygrad.special_token_ids'],[0,1,2])
    self.assertEqual(kv['tokenizer.ggml.scores'],[-float(i) for i in range(len(pieces))])
    tokenizer = SimpleTokenizer.from_gguf_kv(kv)
    ids = tokenizer.encode('a7\na')
    self.assertEqual(ids,[259,260,261,262,260])
    self.assertEqual(tokenizer.decode(ids),'a7\na')
    self.assertEqual(tokenizer.decode([263]),'')

  def test_unsupported_normalization_or_added_embedding_is_rejected(self):
    for options in ({'algorithm':1},{'normalization':'nmt_nfkc'},{'remove_spaces':True},{'byte_fallback':False}):
      with self.subTest(options=options),self.assertRaises(ValueError):tokenizer_metadata(self.model(**options)[0],{})
    data,_ = self.model()
    for config in ({'additional_special_tokens':['<absent>']},
                   {'added_tokens_decoder':{'999':{'content':'<new>','special':True}}},
                   {'added_tokens_decoder':{'1':{'content':'<wrong>','special':True}}}):
      with self.subTest(config=config),self.assertRaises(ValueError):tokenizer_metadata(data,config)

  def test_wire_reader_accepts_unknown_fields_and_refuses_truncation(self):
    data,_ = self.model()
    original,extended = tokenizer_metadata(data,{}),tokenizer_metadata(data+message(100,b'ignored'),{})
    self.assertNotEqual(original.pop('tokenizer.tinygrad.source_sha256'),extended.pop('tokenizer.tinygrad.source_sha256'))
    self.assertEqual(original,extended)
    for invalid in (b'\x00',b'\x80',b'\x08'+b'\xff'*10,b'\x0a\x04ab',b'\x09abc',b'\x0b'):
      with self.subTest(invalid=invalid),self.assertRaises(ValueError):fields(invalid)
