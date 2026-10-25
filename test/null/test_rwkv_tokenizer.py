import unittest
from tinygrad.llm.cli import SimpleTokenizer, RWKVTokenizer


class TestRWKVTokenizer(unittest.TestCase):
  def tokenizer(self):
    values = [bytes([i]) for i in range(256)]+[b'ab',b'abc',b'bc',b"it's",b'\n\n',b'\\n','中文'.encode()]
    tokens = ['<s>']+[repr(value)[2:-1] for value in values]+['[PAD264]']
    return SimpleTokenizer.from_gguf_kv({'tokenizer.ggml.model':'rwkv','tokenizer.ggml.tokens':tokens,
                                       'tokenizer.ggml.token_type':[3]+[1]*len(values)+[5],
                                       'tokenizer.ggml.bos_token_id':0,'tokenizer.ggml.eos_token_id':0,'tokenizer.ggml.eot_token_id':261})

  def test_longest_byte_prefix_and_literal_escape_decoding(self):
    tok = self.tokenizer()
    self.assertIsInstance(tok,RWKVTokenizer)
    for text,expected in [('abcab',[258,257]),('\n\n',[261]),('\\n',[262]),('中文',[263]),('abc\x00',[258,1])]:
      with self.subTest(text=text):
        self.assertEqual(tok.encode(text),expected)
        self.assertEqual(tok.decode(expected),text)
    self.assertNotIn(264,tok.encode('[PAD264]'))

  def test_special_tokens_and_incremental_utf8(self):
    tok = self.tokenizer()
    marker = '<|rwkv_tokenizer_end_of_text|>'
    self.assertEqual(tok.encode(marker+'abc'),[0,258])
    self.assertEqual(tok.decode([0]),marker)
    self.assertIsNone(tok.bos_id)
    self.assertTrue(tok.is_end(0))
    self.assertTrue(tok.is_end(261))
    decode = tok.stream_decoder()
    self.assertEqual(decode(0xe4+1),'')
    self.assertEqual(decode(0xb8+1),'')
    self.assertEqual(decode(0xad+1),'中')
