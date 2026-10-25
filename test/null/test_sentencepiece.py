import unittest
from tinygrad.llm.cli import SimpleTokenizer, SentencePieceTokenizer


class TestSentencePiece(unittest.TestCase):
  def fixture(self):
    tokens = ['<unk>','<s>','</s>']+[f'<0x{i:02X}>' for i in range(256)]+['▁','a','b','c','▁a','ab','bc','▁ab','abc','▁abc']
    return {'tokenizer.ggml.model':'llama','tokenizer.ggml.tokens':tokens,'tokenizer.ggml.token_type':[2,3,3]+[6]*256+[1]*10,
            'tokenizer.ggml.scores':[0.]*len(tokens),'tokenizer.ggml.bos_token_id':1,'tokenizer.ggml.eos_token_id':2,
            'tokenizer.ggml.merges':['a b','▁ a','▁ ab','ab c','▁ab c']}

  def test_merge_ranks_take_precedence_over_vocabulary_ids(self):
    kv = self.fixture()
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertIsInstance(tok,SentencePieceTokenizer)
    self.assertEqual(tok.encode('ab'),[kv['tokenizer.ggml.tokens'].index('▁ab')])
    self.assertEqual(tok.encode('abc'),[kv['tokenizer.ggml.tokens'].index('▁abc')])

  def test_whitespace_byte_fallback_and_streaming(self):
    tok = SimpleTokenizer.from_gguf_kv(self.fixture())
    for text in ('','ab',' ab','a  b','a\nb','\ta\r\n','中文😀','é','a'*200):
      with self.subTest(text=text):
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids),text)
        decoder = tok.stream_decoder()
        self.assertEqual(''.join(decoder(i) for i in ids)+decoder(),text)
    self.assertEqual(tok.encode('<s>ab</s>'),[1,*tok.encode('ab'),2])
    self.assertEqual(tok.decode([1]),'<s>')

  def test_score_vocabulary_and_unigram_rejection(self):
    kv = self.fixture()
    kv.pop('tokenizer.ggml.merges')
    for i in range(259,len(kv['tokenizer.ggml.tokens'])):kv['tokenizer.ggml.scores'][i]=-i
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertEqual(tok.decode(tok.encode('abc')),'abc')
    kv['tokenizer.ggml.scores'][-1]=-1.25
    with self.assertRaisesRegex(ValueError,'Unigram'):SimpleTokenizer.from_gguf_kv(kv)

  def test_minicpm_end_of_turn_fallback(self):
    kv = self.fixture()
    kv['tokenizer.ggml.tokens'].append('<|im_end|>')
    kv['tokenizer.ggml.token_type'].append(4)
    kv['tokenizer.ggml.scores'].append(0.)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertTrue(tok.is_end(len(kv['tokenizer.ggml.tokens'])-1))
    self.assertTrue(tok.is_end(2))
    self.assertFalse(tok.is_end(1))

  def test_gemma_no_dummy_prefix_preserves_real_leading_whitespace(self):
    kv = self.fixture()
    kv['tokenizer.ggml.add_space_prefix']=False
    tok = SimpleTokenizer.from_gguf_kv(kv)
    for text in ('ab',' ab','   ab','\tab'):
      ids = tok.encode(text)
      self.assertEqual(tok.decode(ids),text)
      decoder = tok.stream_decoder()
      self.assertEqual(''.join(decoder(i) for i in ids)+decoder(),text)

  def test_chatglm_control_aliases_and_role_segment_decoding(self):
    kv = self.fixture()
    base = len(kv['tokenizer.ggml.tokens'])
    kv['tokenizer.ggml.pre'] = 'chatglm-spm'
    kv['tokenizer.ggml.tokens'] += ['<|user|>','<|assistant|>','<|observation|>']
    kv['tokenizer.ggml.token_type'] += [3,3,3]
    kv['tokenizer.ggml.scores'] += [0.]*3
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertNotEqual(tok.encode('<unk>'),[0])
    self.assertEqual(tok.decode(tok.encode('<unk>')),'<unk>')
    self.assertEqual(tok.decode([0]),' ⁇ ')
    self.assertEqual(tok.decode([1,2]),'')
    for text in ('ab<|assistant|>ab','<|user|>\nab<|assistant|>','ab<|observation|> ab'):
      ids = tok.encode(text)
      self.assertEqual(tok.decode(ids),text)
      decoder = tok.stream_decoder()
      self.assertEqual(''.join(decoder(i) for i in ids)+decoder(),text)
    self.assertTrue(tok.is_end(base))
    self.assertFalse(tok.is_end(base+1))
    self.assertTrue(tok.is_end(base+2))

  def test_gemma4_routes_bpe_and_ends_at_eos_or_end_turn(self):
    kv = self.fixture()
    end_turn = len(kv['tokenizer.ggml.tokens'])
    kv.update({'tokenizer.ggml.model':'gemma4','tokenizer.ggml.add_space_prefix':False,
               'tokenizer.ggml.bos_token_id':2,'tokenizer.ggml.eos_token_id':1})
    kv['tokenizer.ggml.tokens'].append('<turn|>')
    kv['tokenizer.ggml.token_type'].append(3)
    kv['tokenizer.ggml.scores'].append(0.)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertIsInstance(tok,SentencePieceTokenizer)
    self.assertTrue(tok.is_end(1))
    self.assertTrue(tok.is_end(end_turn))
    self.assertFalse(tok.is_end(2))
    for text in ('ab',' ab','a  b\n中文'):
      self.assertEqual(tok.decode(tok.encode(text)),text)

  def test_user_symbols_do_not_restart_dummy_prefix_or_merge_with_neighbors(self):
    kv = self.fixture()
    base = len(kv['tokenizer.ggml.tokens'])
    kv['tokenizer.ggml.tokens'] += ['\n','\t','<raw>','▁\n','\na']
    kv['tokenizer.ggml.token_type'] += [4,4,4,1,1]
    kv['tokenizer.ggml.scores'] += [0.]*5
    kv['tokenizer.tinygrad.special_token_ids'] = [0,1,2]
    kv['tokenizer.ggml.merges'] += ['▁ \n','\n a']
    tok = SimpleTokenizer.from_gguf_kv(kv)
    prefix = kv['tokenizer.ggml.tokens'].index('▁')
    ab = kv['tokenizer.ggml.tokens'].index('▁ab')
    plain_ab = kv['tokenizer.ggml.tokens'].index('ab')
    self.assertEqual(tok.encode('\nab'),[prefix,base,plain_ab])
    self.assertEqual(tok.encode('ab\nab'),[ab,base,plain_ab])
    self.assertEqual(tok.encode('ab\t<raw>ab'),[ab,base+1,base+2,plain_ab])
    self.assertEqual(tok.decode(tok.encode('ab\nab')),'ab\nab')

  def test_unregistered_control_is_not_parsed_as_an_added_token(self):
    kv = self.fixture()
    control = len(kv['tokenizer.ggml.tokens'])
    kv['tokenizer.ggml.tokens'].append('<reserved>')
    kv['tokenizer.ggml.token_type'].append(3)
    kv['tokenizer.ggml.scores'].append(0.)
    kv['tokenizer.tinygrad.special_token_ids'] = [0,1,2]
    tok = SimpleTokenizer.from_gguf_kv(kv)
    self.assertNotIn(control,tok.encode('<reserved>'))
    self.assertEqual(tok.decode([control,*tok.encode('ab')]),'ab')
    decode = tok.stream_decoder()
    self.assertEqual(''.join(decode(i) for i in [control,*tok.encode('ab')])+decode(),'ab')
    with self.assertRaises(ValueError):SimpleTokenizer.from_gguf_kv({**kv,'tokenizer.tinygrad.special_token_ids':[len(kv['tokenizer.ggml.tokens'])]})
