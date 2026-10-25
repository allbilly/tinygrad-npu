from __future__ import annotations
import sys, argparse, codecs, heapq, itertools, typing, re, unicodedata, json, time, string
from typing import TYPE_CHECKING
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, Context, fetch, profile_marker, getenv, round_up
from tinygrad.llm.model import Transformer
if TYPE_CHECKING:
  import jinja2

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None, eom_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2","lfm2":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4","gpt-4o","smollm","spark2_5","deepseek-v3"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # each limit is one past the category's max codepoint (Z→U+3000, N→U+1FBF9, L→U+323AF, M→U+E01EF)
    # compact adjacent codepoints into ranges: listing them all makes re spend seconds on large prompts
    def ucat_range(pre:str|tuple[str, ...]) -> str:
      limits = {"Z": 0x3001, "N": 0x1fbfa, "L": 0x323b0, "M": 0xe01f0, "P": 0x110000, "S": 0x110000}
      limit = max(limits[p if p in limits else p[0]] for p in (pre if isinstance(pre, tuple) else (pre,)))
      cps = enumerate(cp for cp in range(limit) if unicodedata.category(chr(cp)).startswith(pre))
      runs = [list(g) for _, g in itertools.groupby(cps, lambda e: e[1]-e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    contr, r_l, r_n = "(?i:'s|'t|'re|'ve|'m|'ll|'d)", f"[^\\r\\n{r_p_N}{r_p_L}]?", f"[{r_p_N}]" if preset == "tekken" else f"[{r_p_N}]{{1,3}}"
    r_p, r_w, r_t = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*", f"{contr}|{r_l}[{r_p_L}]+", f"[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    if preset in ("tekken", "gpt-4o"):
      r_up, r_lo = ucat_range(("Lu","Lt","Lm","Lo","M")), ucat_range(("Ll","Lm","Lo","M"))
      sfx = f"{contr}?" if preset == "gpt-4o" else ""
      r_p, r_w = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n/]*", f"{r_l}[{r_up}]*[{r_lo}]+{sfx}|{r_l}[{r_up}]+[{r_lo}]*{sfx}"
    if preset == "smollm":
      # SmolLM2: Digits(individual_digits=True), then GPT-2 ByteLevel(use_regex=True).
      # Digits split before ByteLevel, so the preceding space is its own token.
      r_w = f"'s|'t|'re|'ve|'m|'ll|'d| ?[{r_p_L}]+"
      r_n, r_p, r_t = f"[{r_p_N}]", f" ?[^{r_ws}{r_p_N}{r_p_L}]+", f"[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    self._split_to_word = re.compile(f"{r_w}|{r_n}|{r_p}|{r_t}")
    self._pre_splits:tuple[re.Pattern,...] = ()
    if preset in ('spark2_5','deepseek-v3'):
      # Both isolate digits and CJK before their word split. Spark additionally
      # separates individual digits; DeepSeek groups newline runs with punctuation.
      r_p_P,r_p_S,r_p_M = (ucat_range(category) for category in ('P','S','M'))
      tail,newlines = ('',r'[\r\n]') if preset=='spark2_5' else (r'[\r\n]*',f'[{r_ws}]*[\\r\\n]+')
      words = (f'[{re.escape(string.punctuation)}][A-Za-z]+|[^\\r\\n{r_p_L}{r_p_P}{r_p_S}]?[{r_p_L}{r_p_M}]+|'
               f' ?[{r_p_P}{r_p_S}]+{tail}|{newlines}|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+')
      patterns:tuple[str,...] = (f'[{r_p_N}]{{1,3}}',r'[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+',words)
      if preset=='spark2_5':patterns+=(f'[{r_p_N}]',)
      self._pre_splits = tuple(re.compile('('+pattern+')') for pattern in patterns)
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id, self.eom_id = bos_id, eos_id, eot_id, eom_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    if kv.get('tokenizer.ggml.model')=='rwkv':return RWKVTokenizer(kv)
    if kv.get('tokenizer.ggml.model') in ('llama','gemma4'):return SentencePieceTokenizer(kv)
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] not in (2, 3, 4)) # UNKNOWN|CONTROL|USER_DEFINED
    special_tokens_dict = dict(special_tokens)
    return SimpleTokenizer(dict(normal_tokens), special_tokens_dict, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_tokens_dict.get('<|im_end|>')),
      eom_id=kv.get('tokenizer.ggml.eom_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    words = [chunk] if self._pre_splits else self._split_to_word.findall(chunk)
    for pattern in self._pre_splits:words = [piece for word in words for piece in pattern.split(word) if piece]
    return [tok for word in words for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id, self.eom_id)

class RWKVTokenizer(SimpleTokenizer):
  """RWKV World longest-byte-prefix tokenizer; GGUF stores repr(bytes)[2:-1]."""
  def __init__(self, kv:dict):
    self.preset = 'rwkv'
    self.bos_id = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token',False) else None
    self.eos_id,self.eot_id,self.eom_id = kv.get('tokenizer.ggml.eos_token_id',0),kv.get('tokenizer.ggml.eot_token_id',261),None
    self._normal_tokens,self._tok2bytes = {},{}
    self._trie:dict = {}
    for index,(text,kind) in enumerate(zip(kv['tokenizer.ggml.tokens'],kv['tokenizer.ggml.token_type'])):
      value:bytes = typing.cast(bytes,codecs.escape_decode(text.encode('ascii'))[0]) if kind==1 else text.encode()
      self._tok2bytes[index] = value
      if kind!=1:continue
      if not value:raise ValueError('Empty RWKV vocabulary token')
      self._normal_tokens[value] = index
      node = self._trie
      for byte in value:node = node.setdefault(byte,{})
      node[-1] = index
    # The official HF tokenizer/template names token zero this way; GGUF uses <s>.
    marker = '<|rwkv_tokenizer_end_of_text|>'
    self._special_tokens = {marker:0}
    self._tok2bytes[0] = marker.encode()
    self._split_to_sentence = re.compile(re.escape(marker))

  def _encode_sentence(self, chunk:str) -> list[int]:
    data,offset,ids = chunk.encode(),0,[]
    while offset<len(data):
      node,index,best,end = self._trie,offset,None,offset
      while index<len(data) and data[index] in node:
        node = node[data[index]]
        index+=1
        if -1 in node:best,end = node[-1],index
      if best is None:raise ValueError('RWKV vocabulary cannot encode this byte')
      ids.append(best)
      offset = end
    return ids


class SentencePieceTokenizer(SimpleTokenizer):
  """SentencePiece BPE with GGUF merge ranks or piece scores and UTF-8 byte fallback.

  This handles Llama/MiniCPM BPE vocabularies, not SentencePiece Unigram models.
  BOS insertion belongs to the chat template, matching SimpleTokenizer's API.
  """
  def __init__(self, kv:dict):
    tokens,types = kv['tokenizer.ggml.tokens'],kv['tokenizer.ggml.token_type']
    self.preset = 'sentencepiece'
    self.bos_id = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token',True) else None
    self._sp_vocab = {s:i for i,s in enumerate(tokens) if types[i]==1}
    self._sp_segmented = kv.get('tokenizer.ggml.pre')=='chatglm-spm'
    registered = kv.get('tokenizer.tinygrad.special_token_ids')
    if registered is None and self._sp_segmented:registered = [i for i,kind in enumerate(types) if i>2 and kind in (3,4)]
    if registered is not None and any(type(i) is not int or not 0<=i<len(tokens) for i in registered):
      raise ValueError('Invalid registered SentencePiece special token IDs')
    allowed = set(registered) if registered is not None else None
    self._special_tokens = {s:i for i,s in enumerate(tokens) if types[i] in (2,3,4) and (allowed is None or i in allowed)}
    self._sp_special_ids = set(self._special_tokens.values())
    self._sp_prefix_ids = {i for i,s in enumerate(tokens) if types[i]==1 and s.startswith('▁')} if self._sp_segmented else set()
    self._sp_user = {s:i for i,s in enumerate(tokens) if types[i]==4 and s not in self._special_tokens}
    self._sp_user_pattern = re.compile('|'.join(re.escape(s) for s in sorted(self._sp_user,key=len,reverse=True))) if self._sp_user else None
    self.eos_id,self.eom_id = kv.get('tokenizer.ggml.eos_token_id',2),kv.get('tokenizer.ggml.eom_token_id')
    self.eot_id = kv.get('tokenizer.ggml.eot_token_id',self._special_tokens.get('<|im_end|>'))
    if kv.get('tokenizer.ggml.model')=='gemma4':
      self.eot_id = kv.get('tokenizer.ggml.eot_token_id',self._special_tokens.get('<turn|>'))
    if self._sp_segmented:
      self.eot_id = kv.get('tokenizer.ggml.eot_token_id',self._special_tokens.get('<|user|>'))
      self.eom_id = kv.get('tokenizer.ggml.eom_token_id',self._special_tokens.get('<|observation|>'))
    self._sp_bytes = {int(s[3:5],16):i for i,s in enumerate(tokens) if types[i]==6 and re.fullmatch(r'<0x[0-9A-Fa-f]{2}>',s)}
    self._sp_scores = kv.get('tokenizer.ggml.scores',[])
    self._sp_merges = {tuple(s.split(' ',1)):i for i,s in enumerate(kv.get('tokenizer.ggml.merges',[]))}
    if not self._sp_merges and (len(self._sp_scores)!=len(tokens) or
                               any(self._sp_scores[i]!=int(self._sp_scores[i]) for i in self._sp_vocab.values())):
      raise ValueError('SentencePiece Unigram tokenization is not supported')
    self._sp_prefix = kv.get('tokenizer.ggml.add_space_prefix',True)
    self._tok2bytes = {i:(b'' if types[i]==3 and allowed is not None and i not in allowed else
                         bytes([int(s[3:5],16)]) if types[i]==6 and re.fullmatch(r'<0x[0-9A-Fa-f]{2}>',s)
                         else s.replace('▁',' ').encode()) for i,s in enumerate(tokens)}
    if self._sp_segmented:
      for i,kind in enumerate(types):
        if kind==2:self._tok2bytes[i] = kv.get('tokenizer.tinygrad.unknown_surface',' ⁇ ').encode()
    self._split_to_sentence = re.compile('|'.join(re.escape(s) for s in self._special_tokens) or r'(?!)')

  def _encode_sentence(self, chunk:str) -> list[int]:
    if not chunk:return []
    pieces,protected = (['▁'],[False]) if self._sp_prefix else ([],[])
    def append_text(text:str):
      chars = list(text.replace(' ','▁'))
      pieces.extend(chars)
      protected.extend([False]*len(chars))
    if self._sp_user_pattern is not None:
      start = 0
      for match in self._sp_user_pattern.finditer(chunk):
        append_text(chunk[start:match.start()])
        pieces.append(match.group())
        protected.append(True)
        start = match.end()
      append_text(chunk[start:])
    else:append_text(chunk)
    previous,next_ = list(range(-1,len(pieces)-1)),list(range(1,len(pieces)+1))
    merges:list[tuple[float,int,int,str,str]] = []
    def offer(left:int):
      if left<0 or (right:=next_[left])>=len(pieces):return
      if protected[left] or protected[right]:return
      pair = pieces[left],pieces[right]
      if self._sp_merges:
        if pair not in self._sp_merges:return
        score = -self._sp_merges[pair]
      else:
        if (token:=self._sp_vocab.get(pair[0]+pair[1])) is None:return
        score = self._sp_scores[token]
      heapq.heappush(merges,(-score,left,right,*pair))
    for i in range(len(pieces)):offer(i)
    while merges:
      _,left,right,a,b = heapq.heappop(merges)
      if next_[left]!=right or pieces[left]!=a or pieces[right]!=b:continue
      pieces[left],pieces[right] = a+b,''
      next_[left] = next_[right]
      if next_[right]<len(pieces):previous[next_[right]]=left
      offer(previous[left])
      offer(left)
    result = []
    for piece in pieces:
      if not piece:continue
      if piece in self._sp_user:result.append(self._sp_user[piece])
      elif piece in self._sp_vocab:result.append(self._sp_vocab[piece])
      else:
        try:result.extend(self._sp_bytes[b] for b in piece.encode())
        except KeyError as exc:raise ValueError('SentencePiece byte fallback token is missing') from exc
    return result

  def decode(self, ids:list[int]) -> str:
    if self._sp_segmented:
      decoder = self.stream_decoder()
      return ''.join(decoder(i) for i in ids)+decoder()
    text = b''.join(self._tok2bytes[i] for i in ids).decode(errors='replace')
    return text.removeprefix(' ') if self._sp_prefix else text

  def stream_decoder(self) -> typing.Callable[..., str]:
    decoder,first = codecs.getincrementaldecoder('utf-8')('replace'),True
    def decode(tid:int|None=None) -> str:
      nonlocal first
      if self._sp_segmented and tid in self._sp_special_ids:
        text = decoder.decode(b'',final=True)
        decoder.reset()
        first = True
        return text+self._tok2bytes[tid].decode()
      data = self._tok2bytes[tid] if tid is not None else b''
      if tid is not None and first and data:
        if self._sp_prefix and (not self._sp_segmented or tid in self._sp_prefix_ids):data=data.removeprefix(b' ')
        first=False
      return decoder.decode(data,final=tid is None)
    return decode

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.6:27b": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf",
  "qwen3.6:35b-a3b": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  # pinned to the last revision with the plain IQ4_XS quant (the replacement uses mixed UD quantization)
  "qwen3.8:27b": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/b62a80264f8b0c1bb849ee1c9c487415ebeca194/Qwen3.8-27B-IQ4_XS.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

class FallbackTemplate:
  # minimal jinja2.Template-compatible chat template without jinja2, no tool calling support
  def __init__(self, tok:SimpleTokenizer): self.tok = tok
  def role(self, role:str) -> str:
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"  # OLMoE Instruct format
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset == 'qwen2': return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self) -> str:
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset == 'qwen2': return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True, preserve_thinking:bool=False,
             enable_thinking:bool=False) -> str:
    # Thinking flags do not change this minimal fallback's rendered content.
    out = self.tok.decode([] if self.tok.bos_id is None else [self.tok.bos_id]) + ("<sop>" if self.tok.preset == 'glm4' else "")
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

from tinygrad.llm.serve import LLMServer

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=lambda s: tuple(int(x) for x in (s if "," in s else f"0,{s}").split(",", 1)), const=(0, 20),
                      metavar="[PREFILL,]DECODE", help="Benchmark tok/s (PREFILL prompt tokens then DECODE new tokens, default 0,20)")
  parser.add_argument("--no_chat_template", action="store_true", help="Don't use the model's chat template, always use the fallback template")
  parser.add_argument("--shard", type=int, default=1, help="Tensor parallel device count")
  parser.add_argument("--gpu-q8-projections", action="store_true", help="Experimentally keep direct Qwen body projections in Q8_0 on Mali OpenCL")
  args = parser.parse_args()
  if args.benchmark: args.max_context = max(args.max_context, round_up(sum(args.benchmark)+1, 1024))

  # load the model
  st = time.perf_counter()
  with Context(DEBUG=max(DEBUG.value, 1 if args.serve else 0)):
    checkpoint = fetch(models.get(args.model, args.model))
    model, kv = Transformer.from_gguf(checkpoint, args.max_context, shard=args.shard)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes()*args.shard for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"loaded model \"{model_name}\" at {sum(file_sizes)*1e-9/(time.perf_counter()-st):.2f} GB/s with {sum(file_sizes):,} bytes "
        f"and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params, "
        f"max context {args.max_context} on {nn.state.get_parameters(model)[0].device}")
  if args.gpu_q8_projections:
    from tinygrad.llm.kernels.mali import configure
    print(f"Mali prepared operations: {json.dumps(configure(model,q8_projections_checkpoint=checkpoint))}")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if not args.no_chat_template and (ct := kv.get('tokenizer.chat_template')) is not None:
    try:
      import jinja2
      env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])  # (for {% break %})
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # warmup the JIT
  if args.warmup or args.serve:
    with Context(DEBUG=max(DEBUG.value, 1)): model.warmup()

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, template).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    keep, (n_pp, n_tg) = 20, args.benchmark
    gen = model.generate(toks:=tok.encode(" ".join(map(str, range(n_pp))))[:n_pp] if n_pp else [tok.bos_id or 0])
    for i in range(n_tg+1):
      profile_marker(f"decode @ {i-1}" if i else "prefill")
      if i == keep+1 and n_tg > 2*keep: print("...")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {(max(n_pp, 1) if i == 0 else 1)*1e9/x:8.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks[n_pp if i <= keep or n_tg <= 2*keep else max(n_pp, 1)+n_tg-keep+1:]).replace("\n", "\\n"),
                  prefix=f"prefill {max(n_pp, 1):5d}: " if i == 0 else f"decode {i-1:6d}: ", enabled=i <= keep or i > n_tg-keep):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
    exit(0)

  # interactive chat
  messages: list[dict] = []
  while 1:
    try: messages.append({"role":"user", "content":input('>>> ')})
    except EOFError: break
    ids = tok.encode(template.render(messages=messages, add_generation_prompt=True))
    reply, dec = "", tok.stream_decoder()
    for next_id in model.generate(ids):
      if tok.is_end(next_id):
        sys.stdout.write(dec() + "\n\n")
        break
      reply += (piece := dec(next_id))
      sys.stdout.write(piece)
      sys.stdout.flush()
    messages.append({"role":"assistant", "content":reply})

if __name__ == "__main__": main()
