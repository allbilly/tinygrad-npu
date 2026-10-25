"""CPU-only regression fixtures for measured-cost speculative fallback.

The coordinator is exercised with independent causal CPU models from
``test.models.test_speculative``. No checkpoint, service, or device is used.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from test.models.test_speculative import CausalTarget, FixedDraft, OracleDraft, oracle, ordinary_ids, real_pair
from tinygrad.llm.speculative import GreedySpeculativeDecoder


class TestMeasuredCostFallback(unittest.TestCase):
    def test_early_anchor_and_fallback_share_the_committed_prefix(self):
        target,draft = CausalTarget(),OracleDraft(wrong_at=0)
        decoder = GreedySpeculativeDecoder(target,draft,early_anchor=True)
        prompt = [2,5,3]
        output = list(decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12))
        self.assert_causal_state(target,prompt,output)
        self.assertTrue(decoder.stats.early_anchor)
        self.assertGreater(decoder.stats.fallback_tokens,0)
        stream = decoder.generate(prompt,max_new_tokens=12,ordinary_decode_seconds=1e-12)
        first = next(stream)
        stream.close()
        self.assert_causal_state(target,prompt,[first])
        self.assertEqual(decoder.stats.verify_calls,0)

    @staticmethod
    def make(context: int = 64):
        target = CausalTarget(context)
        draft = OracleDraft()
        decoder = GreedySpeculativeDecoder(target, draft)
        return target, draft, decoder

    def assert_causal_state(self, target, prompt: list[int], generated: list[int]) -> None:
        expected, state, history = oracle(prompt, len(generated))
        self.assertEqual(generated, expected)
        self.assertEqual(target._cached_tokens, prompt + generated)
        self.assertEqual(int(target.blk[0].recurrent_state.item()), state)
        self.assertEqual(tuple(target.blk[0].conv_state.tolist()[0]), history)
        self.assertEqual(target.blk[0].cache_kv.numpy()[:len(prompt + generated)].tolist(), prompt + generated)

    @staticmethod
    def instrument_causal_target(target):
        calls = []
        original = target.target_forward

        def traced(tokens, start_pos, **kwargs):
            ids = list(tokens.tolist()[0])
            calls.append((start_pos, ids, kwargs.get('last_only', False)))
            return original(tokens, start_pos, **kwargs)

        target.target_forward = traced
        return calls

    @staticmethod
    def instrument_draft(draft):
        events = []
        observe, propose = draft.observe, draft.propose

        def traced_observe(output, start_pos):
            events.append(('observe', start_pos))
            return observe(output, start_pos)

        def traced_propose(anchor, *, position, max_tokens):
            events.append(('propose', position))
            return propose(anchor, position=position, max_tokens=max_tokens)

        draft.observe, draft.propose = traced_observe, traced_propose
        return events

    def test_cost_fallback_matches_independent_oracle_and_commits_state_once(self) -> None:
        target, draft, decoder = self.make()
        prompt = [2, 5, 3]
        calls = self.instrument_causal_target(target)
        draft_events = self.instrument_draft(draft)
        stream = decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12)
        output, events_at_fallback = [], None
        for token in stream:
            output.append(token)
            if decoder.stats.fallback_tokens and events_at_fallback is None:
                events_at_fallback = list(draft_events)
                self.assertGreaterEqual(decoder.stats.blocks, 2)
        self.assert_causal_state(target, prompt, output)
        self.assertTrue(decoder.stats.adaptive)
        self.assertEqual(decoder.stats.ordinary_decode_seconds, 1e-12)
        self.assertGreater(decoder.stats.fallback_tokens, 0)
        self.assertGreater(decoder.stats.fallback_calls, 0)
        self.assertGreater(decoder.stats.fallback_seconds, 0.0)
        self.assertEqual(decoder.stats.fallback_reason, 'measured_round_cost')
        self.assertEqual(draft_events, events_at_fallback, 'draft work must stop after the switch')
        self.assertEqual(calls[0][0:2], (0, prompt))
        self.assertEqual(sum(start_pos == 0 for start_pos, _, _ in calls), 1, 'fallback must not re-prefill')
        self.assertEqual([(start, len(ids)) for start, ids, _ in calls[:3]], [(0, 3), (3, 4), (7, 4)])
        fallback_calls = calls[3:]
        self.assertTrue(fallback_calls)
        self.assertTrue(all(len(ids) == 1 for _, ids, _ in fallback_calls), fallback_calls)
        self.assertEqual(fallback_calls[0][0], len(prompt) + 8)
        self.assertEqual(len(fallback_calls), decoder.stats.fallback_calls)

    def test_extra_compute_budget_falls_back_after_one_causal_block(self):
        target, draft, decoder = self.make()
        prompt = [2, 5, 3]
        calls = self.instrument_causal_target(target)
        draft_events = self.instrument_draft(draft)
        events_at_fallback = None
        output = []
        stream = decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                  max_speculative_extra_seconds=1e-12)
        for token in stream:
            output.append(token)
            if decoder.stats.fallback_tokens and events_at_fallback is None:
                events_at_fallback = list(draft_events)
                self.assertEqual(decoder.stats.blocks, 1)
        self.assert_causal_state(target, prompt, output)
        self.assertTrue(decoder.stats.adaptive)
        self.assertEqual(decoder.stats.max_speculative_extra_seconds, 1e-12)
        self.assertEqual(decoder.stats.fallback_reason, 'measured_extra_cost')
        self.assertEqual(decoder.stats.blocks, 1)
        self.assertGreater(decoder.stats.fallback_tokens, 0)
        self.assertIsNotNone(events_at_fallback)
        self.assertEqual(draft_events, events_at_fallback, 'draft work must stop after the first expensive block')
        self.assertEqual(calls[0][0:2], (0, prompt))
        self.assertEqual(sum(start_pos == 0 for start_pos, _, _ in calls), 1)

    def test_extra_compute_budget_requires_ordinary_reference_before_state_or_draft_work(self):
        target, draft, decoder = self.make()
        prompt = [2, 5, 3]
        draft_events = self.instrument_draft(draft)
        with self.assertRaisesRegex(ValueError, 'Extra compute budget requires an ordinary decode reference'):
            list(decoder.generate(prompt, max_new_tokens=4, max_speculative_extra_seconds=0.01))
        self.assertEqual(target._cached_tokens, [])
        self.assertEqual(int(target.blk[0].recurrent_state.item()), 0)
        self.assertEqual(target.blk[0].conv_state.tolist(), [[0, 0]])
        self.assertEqual(draft_events, [])
        self.assertFalse(decoder._lock.locked())

    def test_fallback_stops_on_eos_and_context_limits(self) -> None:
        target, _, decoder = self.make()
        prompt = [2, 5, 3]
        expected, _, _ = oracle(prompt, 12)
        eos_position = next(index for index in range(8, 12) if expected[index] not in expected[:index])
        eos_token = expected[eos_position]
        output = list(decoder.generate(prompt, max_new_tokens=12, eos_token_ids=[eos_token],
                                       ordinary_decode_seconds=1e-12))
        self.assertEqual(output, expected[:eos_position + 1])
        self.assert_causal_state(target, prompt, output)
        self.assertEqual(decoder.finish_reason, 'eos')
        self.assertGreater(decoder.stats.fallback_tokens, 0)

        target, _, decoder = self.make(context=13)
        output = list(decoder.generate(prompt, max_new_tokens=20, ordinary_decode_seconds=1e-12))
        self.assertEqual(len(output), 10)
        self.assert_causal_state(target, prompt, output)
        self.assertEqual(decoder.finish_reason, 'context')
        self.assertGreater(decoder.stats.fallback_tokens, 0)

    def test_close_after_fallback_token_preserves_state_and_decoder_reuses(self) -> None:
        target, _, decoder = self.make()
        prompt = [2, 5, 3]
        stream = decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12)
        output = []
        for token in stream:
            output.append(token)
            if decoder.stats.fallback_tokens:
                break
        self.assertGreater(decoder.stats.fallback_tokens, 0)
        stream.close()
        self.assertEqual(decoder.finish_reason, 'closed')
        self.assert_causal_state(target, prompt, output)
        again_prompt = [7, 8]
        again = list(decoder.generate(again_prompt, max_new_tokens=5))
        self.assert_causal_state(target, again_prompt, again)
        self.assertFalse(decoder.stats.adaptive)
        self.assertIsNone(decoder.stats.fallback_reason or None)

    def test_cancel_during_fallback_forward_rolls_back_before_emitting_and_reuses(self) -> None:
        target, _, decoder = self.make()
        prompt = [2, 5, 3]
        stop = {'value': False}
        original = target.target_forward
        calls = []

        def cancel_after_fallback_compute(tokens, start_pos, **kwargs):
            ids = list(tokens.tolist()[0])
            calls.append((start_pos, ids))
            output = original(tokens, start_pos, **kwargs)
            if len(ids) == 1 and start_pos > len(prompt):
                stop['value'] = True
            return output

        target.target_forward = cancel_after_fallback_compute
        stream = decoder.generate(prompt, max_new_tokens=12, ordinary_decode_seconds=1e-12,
                                  cancelled=lambda: stop['value'])
        output = list(stream)
        self.assertEqual(len(output), 8)
        self.assertEqual(calls[0], (0, prompt))
        self.assertTrue(any(start > len(prompt) and len(ids) == 1 for start, ids in calls))
        self.assertEqual(decoder.finish_reason, 'cancelled')
        self.assertEqual(decoder.stats.fallback_reason, 'measured_round_cost')
        self.assertEqual(decoder.stats.fallback_tokens, 0, 'canceled anchor must not be emitted or counted')
        self.assert_causal_state(target, prompt, output)
        again_prompt = [9, 3]
        again = list(decoder.generate(again_prompt, max_new_tokens=4))
        self.assert_causal_state(target, again_prompt, again)

    def test_invalid_reference_fails_before_lock_or_target_state_write(self) -> None:
        target, draft, decoder = self.make()
        target_calls = self.instrument_causal_target(target)
        draft_events = self.instrument_draft(draft)
        prompt = [2, 5, 3]
        for invalid in (0.0, -1.0, float('nan'), float('inf'), True, '1'):
            with self.subTest(reference=invalid):
                with self.assertRaises(ValueError):
                    list(decoder.generate(prompt, max_new_tokens=3, ordinary_decode_seconds=invalid))
                self.assertEqual(target._cached_tokens, [])
                self.assertEqual(int(target.blk[0].recurrent_state.item()), 0)
                self.assertEqual(target.blk[0].conv_state.tolist(), [[0, 0]])
                self.assertFalse(decoder._lock.locked())
                self.assertEqual(target_calls, [])
                self.assertEqual(draft_events, [])
        output = list(decoder.generate(prompt, max_new_tokens=3))
        self.assert_causal_state(target, prompt, output)

    def test_consumer_pause_does_not_trigger_compute_cost_fallback(self) -> None:
        target, _, decoder = self.make()
        prompt = [2, 5, 3]
        clock = [0.0]

        def synthetic_perf_counter():
            value = clock[0]
            clock[0] += 0.001
            return value

        output = []
        with patch('tinygrad.llm.speculative.perf_counter', side_effect=synthetic_perf_counter):
            stream = decoder.generate(prompt, max_new_tokens=10, ordinary_decode_seconds=1.0)
            for token in stream:
                output.append(token)
                clock[0] += 1000.0  # Consumer-side pause is outside the measured phase deltas.
        self.assert_causal_state(target, prompt, output)
        self.assertTrue(decoder.stats.adaptive)
        self.assertEqual(decoder.stats.fallback_tokens, 0)
        self.assertEqual(decoder.stats.fallback_reason, '')

    def test_hybrid_transformer_fallback_matches_ordinary_cpu_reference(self) -> None:
        reference, target = real_pair(hybrid=True)
        prompt = [1, 2, 3]
        expected = ordinary_ids(reference, prompt, 10)
        draft = FixedDraft(len(prompt), expected)
        decoder = GreedySpeculativeDecoder(target, draft)
        output = list(decoder.generate(prompt, max_new_tokens=len(expected), ordinary_decode_seconds=1e-12))
        self.assertEqual(output, expected)
        self.assertTrue(decoder.stats.adaptive)
        self.assertGreaterEqual(decoder.stats.blocks, 2)
        self.assertGreater(decoder.stats.fallback_tokens, 0)
        self.assertEqual(decoder.stats.fallback_reason, 'measured_round_cost')


if __name__ == '__main__':
    unittest.main()
