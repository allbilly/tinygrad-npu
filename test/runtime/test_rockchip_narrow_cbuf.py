"""CPU-only register recipe checks for the narrow decode CBUF flag."""
import unittest
from unittest.mock import patch

from tinygrad.runtime.support import rockchip_int8 as rk8

FLAG = 'ONPU_INT8_DECODE_NARROW_CBUF'
QUALIFIED_N = (128, 192, 256, 320, 352, 416, 448, 576, 608, 640, 704, 768, 832, 864, 896, 960)
UNCHANGED_N = (32, 64, 96, 288, 384, 480, 544)


def body(flag, m, n, k):
  with patch.object(rk8, 'getenv', side_effect=lambda key, default: flag if key == FLAG else default):
    return rk8.int8_body(m, n, k, 0x100000, 0x200000, 0x300000, n)


def reg(word):
  return word >> 48, word & 0xffff, (word >> 16) & 0xffffffff


class TestNarrowCbufFlag(unittest.TestCase):
  def test_flag_changes_only_two_cbuf_writes_for_qualified_decode_shapes(self):
    for n in QUALIFIED_N:
      with self.subTest(n=n):
        baseline, enabled = body(0, 1, n, 4096), body(1, 1, n, 4096)
        self.assertEqual(len(baseline), len(enabled))
        changes = [(a, b) for a, b in zip(baseline, enabled) if a != b]
        self.assertEqual(len(changes), 2)
        self.assertTrue(all(reg(a)[:2] == (0x201, 0x1040) for a, _ in changes))
        self.assertTrue(all(reg(b) == (0x201, 0x1040, 0xb1) for _, b in changes))

  def test_flag_preserves_baseline_for_nonqualified_and_existing_cases(self):
    for m, n, k in (
      *((1, n, 4096) for n in UNCHANGED_N),
      (1, 512, 4096), (1, 1024, 4096),
      (17, 128, 4096), (64, 128, 4096),
      (1, 704, 1024), (1, 704, 2048),
    ):
      with self.subTest(m=m, n=n, k=k):
        self.assertEqual(body(1, m, n, k), body(0, m, n, k))

  def test_default_704_decode_uses_baseline_cbuf_value(self):
    words = body(0, 1, 704, 4096)
    cbuf = [reg(word)[2] for word in words if reg(word)[:2] == (0x201, 0x1040)]
    self.assertEqual(len(cbuf), 2)
    self.assertEqual(cbuf, [0x48, 0x48])


if __name__ == '__main__':
  unittest.main()
