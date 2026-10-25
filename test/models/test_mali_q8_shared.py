"""Shared Q8 token embedding and tied output head contracts."""
import os, unittest, gc, weakref
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.kernels.mali import MaliEmbedding, MaliLinear
from tinygrad.llm.kernels.mali_q8_shared import MaliQ8Table, MaliQ8Embedding, MaliQ8Head, replace_tied_q8_table
from test.models.test_mali_q8 import q8_fixture


class TestMaliQ8SharedContract(unittest.TestCase):
  """These checks construct lazy CL graphs and must not initialize OpenCL."""
  def test_table_shares_lazy_half_view_and_reports_retained_storage(self):
    raw = Tensor.empty(7 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8)
    table = MaliQ8Table(raw, 7, 96)
    self.assertIs(table.raw, raw)
    self.assertEqual((table.vocab_size, table.dim, table.device), (7, 96, 'CL'))
    self.assertIs(table.weight, table.weight)
    self.assertEqual(table.weight.shape, (7, 96))
    self.assertIs(table.weight.dtype, dtypes.half)
    self.assertFalse(table.weight.uop.is_realized)
    metadata = table.metadata()
    self.assertEqual(metadata['retained_raw_bytes'], raw.numel())
    self.assertEqual(metadata['logical_fp16_bytes'], 7 * 96 * 2)
    self.assertFalse(metadata['logical_weight_realized'])

  def test_embedding_and_head_build_expected_lazy_graphs_and_reject_bad_inputs(self):
    table = MaliQ8Table(Tensor.empty(7 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8), 7, 96)
    embedding = MaliQ8Embedding(table)
    self.assertIs(embedding.table, table)
    self.assertIs(embedding.weight, table.weight)
    ids = Tensor.empty(2, 3, device='CPU', dtype=dtypes.int)
    self.assertEqual(embedding(ids).shape, (2, 3, 96))
    head = MaliQ8Head(table)
    self.assertIs(head.table, table)
    self.assertIs(head.weight, table.weight)
    self.assertEqual((head.in_features, head.out_features, head.bias), (96, 7, None))
    self.assertEqual(head(Tensor.empty(2, 3, 96, device='CL', dtype=dtypes.float)).shape, (2, 3, 7))
    for x in (Tensor.empty(1, 96, device='CL', dtype=dtypes.half),
              Tensor.empty(1, 9, 96, device='CL', dtype=dtypes.float),
              Tensor.empty(1, 96, device='CPU', dtype=dtypes.float)):
      with self.subTest(shape=x.shape, dtype=x.dtype, device=x.device), self.assertRaises(ValueError): head(x)

  def test_table_rejects_invalid_q8_storage_and_dimensions(self):
    for raw, vocab, dim in (
      (Tensor.empty(7 * 96 // 32 * 34, device='CPU', dtype=dtypes.uint8), 7, 96),
      (Tensor.empty(7 * 96 // 32 * 34, device='CL', dtype=dtypes.int8), 7, 96),
      (Tensor.empty(7 * 96 // 32 * 34 - 1, device='CL', dtype=dtypes.uint8), 7, 96),
      (Tensor.empty(7 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8), 7, 95),
    ):
      with self.subTest(vocab=vocab, dim=dim), self.assertRaises(ValueError): MaliQ8Table(raw, vocab, dim)

  def test_replacement_rejects_untied_or_unclean_models_before_mutation(self):
    def fresh_model(tied=True):
      source = nn.Embedding(7, 96)
      source.weight = Tensor.empty(7, 96, device='CL', dtype=dtypes.half)
      words = MaliEmbedding(source)
      output_source = nn.Linear(96, 7, bias=False)
      output_source.weight = Tensor.empty(7, 96, device='CL', dtype=dtypes.half)
      if tied: output_source.weight = source.weight
      output = MaliLinear(output_source, prepare_gemm=False, consistent_decode=True)
      return SimpleNamespace(token_embd=words, output=output, _cached_tokens=[])
    raw = Tensor.empty(7 * 96 // 32 * 34, device='CL', dtype=dtypes.uint8)
    model = fresh_model(tied=False)
    old_words, old_output = model.token_embd, model.output
    with self.assertRaises(ValueError): replace_tied_q8_table(model, raw)
    self.assertIs(model.token_embd, old_words)
    self.assertIs(model.output, old_output)
    model = fresh_model()
    model.blk = [SimpleNamespace(cache_kv=Tensor.empty(1, device='CL'))]
    old_words, old_output = model.token_embd, model.output
    with self.assertRaisesRegex(ValueError, 'before model state allocation'):
      replace_tied_q8_table(model, raw)
    self.assertIs(model.token_embd, old_words)
    self.assertIs(model.output, old_output)
    model = fresh_model()
    model._cached_tokens = [1]
    old_words, old_output = model.token_embd, model.output
    with self.assertRaisesRegex(ValueError, 'before model generation or graph capture'):
      replace_tied_q8_table(model, raw)
    self.assertIs(model.token_embd, old_words)
    self.assertIs(model.output, old_output)
    model = fresh_model()
    model.prefill_jit = TinyJit(lambda x: x)
    model.prefill_jit.cnt = 1
    old_words, old_output = model.token_embd, model.output
    with self.assertRaisesRegex(ValueError, 'before model generation or graph capture'):
      replace_tied_q8_table(model, raw)
    self.assertIs(model.token_embd, old_words)
    self.assertIs(model.output, old_output)


@unittest.skipUnless(os.environ.get('MALI_TEST') == '1', 'explicit OpenCL hardware qualification')
class TestMaliQ8SharedHardware(unittest.TestCase):
  def test_replacement_releases_old_dense_tie_and_shares_owned_raw_table(self):
    from tinygrad.llm.kernels.mali_linear import embedding, gemv_rows
    raw_values, weights = q8_fixture(7, 96, seed=503)
    raw = Tensor(raw_values, device='CL', dtype=dtypes.uint8).realize()

    def make_tied_model():
      source_embedding = nn.Embedding(7, 96)
      old_weight = Tensor(weights.copy(), device='CL', dtype=dtypes.half).realize()
      source_embedding.weight = old_weight
      words = MaliEmbedding(source_embedding)
      source_output = nn.Linear(96, 7, bias=False)
      source_output.weight = old_weight
      head = MaliLinear(source_output, prepare_gemm=False, consistent_decode=True)
      buffer_ref = weakref.ref(old_weight.uop.base.buffer)
      return (SimpleNamespace(token_embd=words, output=head, _cached_tokens=[]),
              weakref.ref(words), weakref.ref(head), weakref.ref(old_weight), buffer_ref)

    model, words_ref, head_ref, weight_ref, buffer_ref = make_tied_model()
    self.assertIs(model.token_embd.weight.uop, model.output.weight.uop)
    self.assertIsNotNone(buffer_ref())
    replaced = replace_tied_q8_table(model, raw)
    gc.collect()
    self.assertIsNone(words_ref())
    self.assertIsNone(head_ref())
    self.assertIsNone(weight_ref())
    self.assertIsNone(buffer_ref())
    table = model.token_embd.table
    self.assertIs(model.output.table, table)
    self.assertIs(table.raw, raw)
    self.assertFalse(replaced['logical_weight_realized'])
    self.assertFalse(table.metadata()['logical_weight_realized'])

    # Build the independent dense reference only after proving the old tied
    # module and its realized allocation have been released.
    dense = Tensor(weights.copy(), device='CL', dtype=dtypes.half).realize()
    ids = Tensor([[0, 6, -1, 7]], device='CL', dtype=dtypes.int).realize()
    expected_lookup = embedding(ids, dense).numpy()
    actual_lookup = model.token_embd(ids).numpy()
    np.testing.assert_array_equal(actual_lookup.view(np.uint32), expected_lookup.view(np.uint32))
    x = Tensor(np.random.default_rng(503).normal(0, .3, (1, 3, 96)).astype(np.float32), device='CL').realize()
    expected_head = gemv_rows(x, dense).numpy()
    actual_head = model.output(x).numpy()
    np.testing.assert_array_equal(actual_head.view(np.uint32), expected_head.view(np.uint32))
    self.assertFalse(table.metadata()['logical_weight_realized'])

    # Accessing the logical view remains a supported, explicit materialization.
    table.weight.realize()
    np.testing.assert_array_equal(table.weight.numpy().view(np.uint16), weights.view(np.uint16))
    self.assertTrue(table.metadata()['logical_weight_realized'])

  def test_embedding_matches_dense_for_invalid_ids_and_tinyjit_rebinds(self):
    from tinygrad.llm.kernels.mali_linear import embedding
    raw_values, weights = q8_fixture(7, 96, seed=501)
    raw, dense = Tensor(raw_values, device='CL', dtype=dtypes.uint8).realize(), Tensor(weights, device='CL', dtype=dtypes.half).realize()
    table = MaliQ8Table(raw, 7, 96)
    layer = MaliQ8Embedding(table)
    for values in (np.array([[0, 6, -1, 7, 100000]], dtype=np.int32), np.array([[3, -4, 2, 2**40, 1]], dtype=np.int64)):
      run = TinyJit(lambda ids: layer(ids).realize())
      ids = Tensor(values, device='CL').realize()
      expected = embedding(ids, dense).numpy()
      for _ in range(3):
        actual = run(ids).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    rows = UOp.variable('shared_q8_dynamic_rows', 1, 8)
    dynamic = TinyJit(lambda x, count: layer(x[:, :count]).realize())
    for count in (1, 3, 8, 3):
      values = np.arange(8, dtype=np.int32).reshape(1, 8) % 7
      ids = Tensor(values, device='CL').realize()
      actual = dynamic(ids, rows.bind(count)).pad_to((1, 8, 96)).numpy()
      expected = embedding(Tensor(values[:, :count].copy(), device='CL').realize(), dense).numpy()
      np.testing.assert_array_equal(actual[:, :count].view(np.uint32), expected.view(np.uint32))
      np.testing.assert_array_equal(actual[:, count:].view(np.uint32), np.zeros_like(actual[:, count:]).view(np.uint32))
    self.assertFalse(table.metadata()['logical_weight_realized'])

  def test_head_rows_match_dense_and_preserve_inputs_without_realizing_view(self):
    from tinygrad.llm.kernels.mali_linear import gemv_rows
    raw_values, weights = q8_fixture(17, 96, seed=502)
    raw_before, weights_before = raw_values.copy(), weights.copy()
    raw, dense = Tensor(raw_values, device='CL').realize(), Tensor(weights, device='CL', dtype=dtypes.half).realize()
    table, head = MaliQ8Table(raw, 17, 96), None
    head = MaliQ8Head(table)
    self.assertFalse(table.metadata()['logical_weight_realized'])
    rng = np.random.default_rng(502)
    for shape in ((1, 1, 96), (1, 3, 96), (1, 8, 96), (2, 3, 96)):
      values = rng.normal(0, .3, shape).astype(np.float32)
      x = Tensor(values, device='CL').realize()
      actual, expected = head(x).numpy(), gemv_rows(x, dense).numpy()
      np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    self.assertFalse(table.metadata()['logical_weight_realized'])
    np.testing.assert_array_equal(raw.numpy(), raw_before)
    np.testing.assert_array_equal(dense.numpy().view(np.uint16), weights_before.view(np.uint16))


if __name__ == '__main__': unittest.main()
