from types import SimpleNamespace

import pytest

from tinygrad.llm.gemma4 import Gemma4Block
from tinygrad.llm.kernels.rockchip_shared_pool import gemma4_pool_eligible


def fixture():
  blocks = []
  for i in range(35):
    block = object.__new__(Gemma4Block)
    block.side_dim = 256
    block.config = SimpleNamespace(dim=1536, n_heads=8, n_kv_heads=1, head_dim=512 if i % 5 == 4 else 256)
    blocks.append(block)
  return SimpleNamespace(blk=blocks, max_context=512)


def test_qualified_gemma4_pool():
  assert gemma4_pool_eligible(fixture(), 'on', 'on', 35, 35, 71, True)


@pytest.mark.parametrize('field,value', [('max_context', 1024), ('side_dim', 128), ('dim', 2048),
                                       ('n_heads', 16), ('n_kv_heads', 2), ('head_dim', 128)])
def test_unqualified_geometry_rejected(field, value):
  model = fixture()
  target = model if field == 'max_context' else model.blk[0] if field == 'side_dim' else model.blk[0].config
  setattr(target, field, value)
  assert not gemma4_pool_eligible(model, 'on', 'on', 35, 35, 71, True)


@pytest.mark.parametrize('args', [('off', 'on', 35, 35, 71, True), ('on', 'off', 35, 35, 71, True),
                                 ('on', 'on', 34, 35, 71, True), ('on', 'on', 35, 34, 71, True),
                                 ('on', 'on', 35, 35, 70, True), ('on', 'on', 35, 35, 71, False)])
def test_missing_pooled_preparation_rejected(args):
  assert not gemma4_pool_eligible(fixture(), *args)


def test_partial_or_different_block_family_rejected():
  model = fixture()
  model.blk[0] = SimpleNamespace(**model.blk[0].__dict__)
  assert not gemma4_pool_eligible(model, 'on', 'on', 35, 35, 71, True)
  model = fixture()
  model.blk.pop()
  assert not gemma4_pool_eligible(model, 'on', 'on', 35, 35, 71, True)
