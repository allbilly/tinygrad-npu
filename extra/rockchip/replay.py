"""Reconstruct the blog backend from its pinned base, without changing the working tree.

Pass the blog Markdown and optionally --through-line N for an earlier checkpoint.
--output writes the reconstructed backend; without it, only the replay summary is printed.
Temporary edits which narrow upstream tests are deliberately not applied.
"""
import argparse, subprocess
from pathlib import Path

BASE = "ebf163682acfa4a6be2775c5efa02a503c86a220"
ROOT = Path(__file__).resolve().parents[2]


def apply_block(text:str, block:list[str], line:int) -> str:
  chunks:list[list[str]] = [[]]
  for s in block:
    if s.startswith("@@"): chunks.append([])
    else: chunks[-1].append(s)
  cursor = 0
  for hunk in chunks:
    before = "".join(s[1:] if s.startswith((" ", "-")) else s for s in hunk if not s.startswith("+"))
    after = "".join(s[1:] if s.startswith((" ", "+")) else s for s in hunk if not s.startswith("-"))
    if before == after:
      if (position:=text.find(before, cursor)) >= 0: cursor = position + len(before)
      continue
    if not before: raise ValueError(f"line {line}: insertion has no context")
    position = text.find(before, cursor)
    if position < 0 and text.count(before) == 1: position = text.find(before)
    if position < 0: raise ValueError(f"line {line}: exact context not found: {before[:160]!r}")
    text = text[:position] + after + text[position+len(before):]
    cursor = position + len(after)
  return text


def replay(source:str, through_line:int|None=None) -> tuple[str, int]:
  text = subprocess.check_output(["git", "show", f"{BASE}:tinygrad/runtime/ops_python.py"], cwd=ROOT, text=True)
  lines = source.splitlines(keepends=True)
  limit = min(len(lines), through_line) if through_line is not None else len(lines)
  i = count = 0
  while i < limit:
    if lines[i].startswith("## Full sweep test_ops.py"): break
    if not lines[i].startswith("```"):
      i += 1
      continue
    language, start = lines[i][3:].strip(), i
    i += 1
    block = []
    while i < len(lines) and not lines[i].startswith("```"):
      block.append(lines[i])
      i += 1
    if language == "diff":
      code = "".join(block)
      # These affect test setup or instructions, rather than the backend.
      if not any(marker in code for marker in ("class TestOps", "DO NOT run -n12", "torch.set_default_dtype")):
        text = apply_block(text, block, start+1)
        count += 1
    i += 1
  compile(text, "ops_rockchip.py", "exec")
  return text, count


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("blog", type=Path, help="docs/blog/tinygrad-backend.md from allbilly.github.io")
  parser.add_argument("--through-line", type=int)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  backend, count = replay(args.blog.read_text(), args.through_line)
  if args.output is not None:
    if args.output.resolve() == ROOT/"tinygrad/runtime/ops_rockchip.py":
      parser.error("choose an output outside the live runtime; the committed runtime includes lint/type cleanup")
    args.output.write_text(backend)
  print(f"Replayed {count} backend diff blocks from {BASE[:7]}; reconstructed Python compiles.")
