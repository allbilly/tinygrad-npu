"""Prepared target graph cleanup after synchronization and allocator failures."""
from types import SimpleNamespace
import unittest, weakref
from unittest.mock import Mock, patch
from tinygrad.llm.speculative import GreedySpeculativeDecoder, _PreparedTargetCall, _PreparedTargetForward


class Payload:pass


class Capture:
  def __init__(self, error=None):
    self.ret = Payload()
    self.free_intermediates = Mock(side_effect=error)


def fixture(free_error=None):
  target = SimpleNamespace(max_context=8,token_embd=SimpleNamespace(weight=SimpleNamespace(device='CPU')))
  call = _PreparedTargetCall(target,(),'verify',1,True,False)
  capture = Capture(free_error)
  call.jit.captured = capture
  return call,weakref.ref(capture),weakref.ref(capture.ret),capture.free_intermediates


class TestPreparedTargetCleanup(unittest.TestCase):
  def assert_closed(self, call, captured, result):
    self.assertIsNone(call.jit.captured)
    self.assertIsNone(call.jit.fxn)
    self.assertEqual(call.jit.cnt,0)
    self.assertIsNone(captured())
    self.assertIsNone(result())
    with self.assertRaisesRegex(RuntimeError,'evicted'):call([1],0)

  def test_sync_failure_releases_graph_and_results_even_with_retained_traceback(self):
    for error in (RuntimeError('synchronization failed'),KeyboardInterrupt('cancelled synchronization')):
      with self.subTest(error=type(error).__name__):
        call,captured,result,free = fixture()
        traceback = None
        with patch('tinygrad.llm.speculative._synchronize',side_effect=error) as sync, \
             patch('tinygrad.llm.speculative.Device') as device:
          try:call.close()
          except BaseException as raised:
            self.assertIs(raised,error)
            traceback = raised.__traceback__
          else:self.fail('The original synchronization error was suppressed')
          self.assertIsNotNone(traceback)
          self.assert_closed(call,captured,result)
          free.assert_not_called()
          device.__getitem__.assert_not_called()
          call.close()  # The failed close is terminal and idempotent.
          self.assertEqual(sync.call_count,1)

  def test_scratch_free_failure_releases_traceback_local_and_preserves_original_error(self):
    error = RuntimeError('scratch free failed')
    call,captured,result,free = fixture(error)
    traceback = None
    with patch('tinygrad.llm.speculative._synchronize') as sync, patch('tinygrad.llm.speculative.Device') as device:
      try:call.close()
      except RuntimeError as raised:
        self.assertIs(raised,error)
        traceback = raised.__traceback__
      else:self.fail('The original scratch cleanup error was suppressed')
      self.assertIsNotNone(traceback)
      self.assert_closed(call,captured,result)
      free.assert_called_once_with()
      device.__getitem__.assert_not_called()
      call.close()
      self.assertEqual(sync.call_count,1)
      free.assert_called_once_with()

  def test_allocator_failure_happens_after_reference_release_and_close_is_idempotent(self):
    error = RuntimeError('allocator free failed')
    call,captured,result,free = fixture()
    with patch('tinygrad.llm.speculative._synchronize') as sync, patch('tinygrad.llm.speculative.Device') as device:
      allocator = device.__getitem__.return_value.allocator
      allocator.free_cache.side_effect = error
      try:call.close()
      except RuntimeError as raised:self.assertIs(raised,error)
      else:self.fail('The original allocator cleanup error was suppressed')
      self.assert_closed(call,captured,result)
      call.close()
      self.assertEqual(sync.call_count,1)
      free.assert_called_once_with()
      allocator.free_cache.assert_called_once_with()

  def test_failed_eviction_drops_closed_entry_and_next_call_can_retry(self):
    call,captured,result,_ = fixture()
    helper = _PreparedTargetForward(call.target,(),dynamic_rows=False,max_graphs=1,max_prefill_graphs=1)
    helper.calls[('verify',1,True)] = call
    error = RuntimeError('eviction synchronization failed')
    with patch('tinygrad.llm.speculative._synchronize',side_effect=error):
      with self.assertRaises(RuntimeError) as caught:helper([2],0,'replay',1,True)
    self.assertIs(caught.exception,error)
    self.assertEqual(len(helper.calls),0)
    self.assert_closed(call,captured,result)
    sentinel = object()
    replacement = Mock(return_value=sentinel)
    with patch('tinygrad.llm.speculative._PreparedTargetCall',return_value=replacement) as create:
      self.assertIs(helper([2],0,'replay',1,True),sentinel)
    create.assert_called_once()
    replacement.assert_called_once_with([2],0)
    self.assertEqual(list(helper.calls),[('replay',1,True)])


class TestSpeculativeFinalDraftCleanup(unittest.TestCase):
  def fixture(self, resets, *, early_anchor=False):
    target = SimpleNamespace(config=SimpleNamespace(rwkv7=(),vocab_size=17),max_context=8,_cached_tokens=[],
                             prefill_chunk_size=lambda value:value)
    draft = SimpleNamespace(block_size=4,capture_layers=(),reset=Mock(side_effect=resets))
    return target,draft,GreedySpeculativeDecoder(target,draft,early_anchor=early_anchor)

  def assert_retry(self, draft, decoder):
    self.assertEqual(decoder.finish_reason,'error')
    self.assertTrue(decoder._lock.acquire(blocking=False))
    decoder._lock.release()
    draft.reset.side_effect = None
    self.assertEqual(list(decoder.generate([1],max_new_tokens=0)),[])
    self.assertEqual(decoder.finish_reason,'length')

  def test_final_reset_failure_reports_error_and_releases_lock_for_retry(self):
    error = RuntimeError('final draft reset failed')
    _,draft,decoder = self.fixture([None,error])
    with self.assertRaises(RuntimeError) as caught:list(decoder.generate([1],max_new_tokens=0))
    self.assertIs(caught.exception,error)
    self.assertEqual(draft.reset.call_count,2)
    self.assert_retry(draft,decoder)

  def test_final_reset_failure_preserves_primary_exception_including_cancellation(self):
    for primary in (RuntimeError('primary request failed'),KeyboardInterrupt('primary cancellation')):
      with self.subTest(primary=type(primary).__name__):
        target,draft,decoder = self.fixture([primary,RuntimeError('secondary reset failed')])
        with self.assertRaises(type(primary)) as caught:list(decoder.generate([1],max_new_tokens=0))
        self.assertIs(caught.exception,primary)
        self.assertEqual(target._cached_tokens,[])
        self.assertEqual(draft.reset.call_count,2)
        self.assert_retry(draft,decoder)

  def test_reset_failure_during_normal_stream_close_is_not_hidden_by_generator_exit(self):
    error = RuntimeError('draft close reset failed')
    target,draft,decoder = self.fixture([None,error],early_anchor=True)
    with patch.object(decoder,'_forward',return_value=(Mock(),[1])):
      stream = decoder.generate([2],max_new_tokens=3)
      self.assertEqual(next(stream),1)
      with self.assertRaises(RuntimeError) as caught:stream.close()
    self.assertIs(caught.exception,error)
    self.assertEqual(target._cached_tokens,[2,1])
    self.assert_retry(draft,decoder)


if __name__=='__main__':unittest.main()
