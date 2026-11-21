#!/usr/bin/env python3
"""Stdlib unit tests for qwen_quality_gate.py; no GPU, model or binary needed.

Run like the other script tests:
    python scripts/test_qwen_quality_gate.py
    python -m unittest discover -s scripts -p "test_*.py" -v

Covered:
* the binary dump layout (header field offsets/size) and a write/read round trip,
* the strict refusals: bad magic, bad version, truncated payload, non-finite
  logits, prompt mismatch, vocab mismatch, tier mismatch, missing candidate,
* the metrics themselves (prefix agreement, identical first 32, KL of identical
  and of deliberately shifted distributions),
* the gate thresholds at exactly 28/32 and 27/32,
* the synthetic prompt-id generator against benchmark_qwen.py.
"""

from __future__ import annotations

import math
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import benchmark_qwen as bq
import qwen_quality_gate as qg


def write_dump(path, *, prompt, generated, prefill_logits, final_logits,
               fmt="bf16", backend="cuda", tier="t16", tag="TEST-SKU",
               vocab=None, version=qg.DUMP_VERSION, magic=qg.DUMP_MAGIC,
               temperature=0.0, seed=7, truncate=0, corrupt_index=None):
    """Write a dump exactly like example_qwen::write_quality_dump."""
    vocab = vocab if vocab is not None else len(prefill_logits)
    header = qg.HEADER.pack(magic, version, vocab, len(prompt), len(generated),
                            temperature, seed, fmt.encode(), backend.encode(),
                            tier.encode(), tag.encode())
    blob = bytearray(header)
    blob += struct.pack("<%dI" % len(prompt), *prompt)
    blob += struct.pack("<%dI" % len(generated), *generated)
    pre = list(prefill_logits)
    fin = list(final_logits)
    if corrupt_index is not None:
        pre[corrupt_index] = float("nan")
    blob += struct.pack("<%df" % len(pre), *pre)
    blob += struct.pack("<%df" % len(fin), *fin)
    if truncate:
        blob = blob[:-truncate]
    with open(path, "wb") as f:
        f.write(blob)


def logits(vocab, peak=0, peak_value=10.0):
    v = [0.0] * vocab
    if 0 <= peak < vocab:
        v[peak] = peak_value
    return v


class DumpFormatTests(unittest.TestCase):
    def test_header_size_and_fields(self):
        self.assertEqual(qg.HEADER.size, 160)
        self.assertEqual(qg.DUMP_MAGIC, b"QWLOGITS")
        self.assertEqual(qg.DUMP_VERSION, 1)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.qlogits")
            prompt = [1, 2, 3]
            generated = [4, 5]
            write_dump(path, prompt=prompt, generated=generated,
                       prefill_logits=[0.5, -0.5, 1.0, 2.0],
                       final_logits=[1.0, 0.0, -1.0, 0.25],
                       fmt="w4", backend="vk", tier="t32", tag="RTX 5090",
                       temperature=0.7, seed=123)
            d = qg.Dump(path)
            self.assertEqual(d.prompt, prompt)
            self.assertEqual(d.generated, generated)
            self.assertEqual(d.vocab, 4)
            self.assertEqual(d.format, "w4")
            self.assertEqual(d.backend, "vk")
            self.assertEqual(d.tier, "t32")
            self.assertEqual(d.device_tag, "RTX 5090")
            self.assertAlmostEqual(d.temperature, 0.7, places=6)
            self.assertEqual(d.seed, 123)

    def test_prefill_only_layout(self):
        # Flight-path dumps carry the prefill distribution only; the reader must
        # accept that and report the final one as absent.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.qlogits")
            write_dump(path, prompt=[1], generated=[2],
                       prefill_logits=[0.5, 1.5], final_logits=[])
            d = qg.Dump(path)
            self.assertFalse(d.has_final)
            self.assertEqual(d.prefill_logits, [0.5, 1.5])
            other = os.path.join(tmp, "b.qlogits")
            write_dump(other, prompt=[1], generated=[2],
                       prefill_logits=[0.5, 1.5], final_logits=[])
            m = qg.compare_dumps(d, qg.Dump(other))
            self.assertIsNone(m["kl_final"])
            summary = qg.evaluate([(d, qg.Dump(other))], "t16", "SKU")
            self.assertIsNone(summary["kl_final"])

    def test_refuses_bad_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.qlogits")
            write_dump(path, prompt=[1], generated=[2],
                       prefill_logits=[0.0, 1.0], final_logits=[0.0, 1.0],
                       version=2)
            with self.assertRaises(qg.GateError):
                qg.Dump(path)
            write_dump(path, prompt=[1], generated=[2],
                       prefill_logits=[0.0, 1.0], final_logits=[0.0, 1.0],
                       magic=b"NOTADUMP")
            with self.assertRaises(qg.GateError):
                qg.Dump(path)

    def test_refuses_truncated_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.qlogits")
            write_dump(path, prompt=[1], generated=[2],
                       prefill_logits=[0.0, 1.0], final_logits=[0.0, 1.0],
                       truncate=4)
            with self.assertRaises(qg.GateError):
                qg.Dump(path)
            write_dump(path, prompt=[1], generated=[2],
                       prefill_logits=[0.0, 1.0], final_logits=[0.0, 1.0],
                       corrupt_index=1)
            with self.assertRaises(qg.GateError):
                qg.Dump(path)


class MetricTests(unittest.TestCase):
    def test_identical_dumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.qlogits")
            b = os.path.join(tmp, "b.qlogits")
            ids = list(range(100, 140))
            lg = logits(32, peak=3)
            write_dump(a, prompt=[1, 2], generated=ids, prefill_logits=lg, final_logits=lg)
            write_dump(b, prompt=[1, 2], generated=ids, prefill_logits=lg, final_logits=lg,
                       fmt="w4")
            m = qg.compare_dumps(qg.Dump(a), qg.Dump(b))
            self.assertEqual(m["prefix_agreement"], 1.0)
            self.assertTrue(m["identical_first_32"])
            self.assertAlmostEqual(m["kl_prefill"], 0.0, places=12)
            self.assertAlmostEqual(m["kl_final"], 0.0, places=12)

    def test_prefix_and_kl(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.qlogits")
            b = os.path.join(tmp, "b.qlogits")
            ref_ids = [10] * 40
            cand_ids = [10] * 20 + [11] * 20
            write_dump(a, prompt=[5], generated=ref_ids,
                       prefill_logits=logits(16, peak=2), final_logits=logits(16, peak=2))
            write_dump(b, prompt=[5], generated=cand_ids,
                       prefill_logits=logits(16, peak=3), final_logits=logits(16, peak=2))
            m = qg.compare_dumps(qg.Dump(a), qg.Dump(b))
            self.assertEqual(m["prefix_agreement_ids"], 20)
            self.assertAlmostEqual(m["prefix_agreement"], 0.5)
            # The first divergence is at index 20, inside the first 32 ids.
            self.assertFalse(m["identical_first_32"])
            self.assertEqual(m["identical_first_32"], cand_ids[:32] == ref_ids[:32])
            self.assertGreater(m["kl_prefill"], 0.0)
            self.assertAlmostEqual(m["kl_final"], 0.0, places=12)

    def test_kl_is_directional_and_close_to_the_closed_form(self):
        # p concentrated on 0, q on 1 with a tiny leak: KL(p||q) is finite and
        # equals -log q_0 in this construction.
        p = [10.0, 0.0]
        q = [0.0, 10.0]
        kl = qg.kl_divergence(p, q)
        # p_0 = q_1 = 1/(1+e^-10) and p_1 = q_0 = 1/(1+e^10), so KL is
        # 10 * (p_0 - p_1) = 10 * tanh(5).
        expected = 10.0 * math.tanh(5.0)
        self.assertAlmostEqual(kl, expected, places=9)
        self.assertAlmostEqual(qg.kl_divergence(p, p), 0.0, places=12)

    def test_prompt_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.qlogits")
            b = os.path.join(tmp, "b.qlogits")
            write_dump(a, prompt=[1], generated=[2], prefill_logits=[0, 1], final_logits=[0, 1])
            write_dump(b, prompt=[9], generated=[2], prefill_logits=[0, 1], final_logits=[0, 1])
            with self.assertRaises(qg.GateError):
                qg.compare_dumps(qg.Dump(a), qg.Dump(b))

    def test_vocab_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.qlogits")
            b = os.path.join(tmp, "b.qlogits")
            write_dump(a, prompt=[1], generated=[2], prefill_logits=[0, 1, 2], final_logits=[0, 1, 2])
            write_dump(b, prompt=[1], generated=[2], prefill_logits=[0, 1], final_logits=[0, 1])
            with self.assertRaises(qg.GateError):
                qg.compare_dumps(qg.Dump(a), qg.Dump(b))


class GateTests(unittest.TestCase):
    def _pair(self, tmp, index, *, prefix_ids, tier="t16", tag="SKU"):
        a = os.path.join(tmp, "ref%02d.qlogits" % index)
        b = os.path.join(tmp, "cand%02d.qlogits" % index)
        lg = logits(8, peak=1)
        write_dump(a, prompt=[index + 1], generated=[7] * 128,
                   prefill_logits=lg, final_logits=lg, tier=tier, tag=tag)
        write_dump(b, prompt=[index + 1], generated=prefix_ids,
                   prefill_logits=lg, final_logits=lg, tier=tier, tag=tag, fmt="w4")
        return qg.Dump(a), qg.Dump(b)

    def test_gate_passes_at_28_of_32_with_90_percent_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = []
            for i in range(32):
                if i < 28:
                    ids = [7] * 128
                else:
                    # Diverge at index 31: the first 32 ids then differ, while the
                    # prefix stays 31/128 = 90.5 % overall, so this exercises the
                    # 28/32 branch and the >= 90 % branch together.
                    ids = [7] * 31 + [8] + [7] * 96
                pairs.append(self._pair(tmp, i, prefix_ids=ids))
            summary = qg.evaluate(pairs, "t16", "SKU", require_full=True)
            self.assertEqual(summary["first32_identical"], 28)
            self.assertEqual(summary["gate"]["verdict"], "pass")
            self.assertGreaterEqual(summary["mean_prefix_agreement"], 0.90)

    def test_gate_fails_at_27_of_32(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = []
            for i in range(32):
                # 27 prompts identical, 5 diverge inside the first 32 ids.
                ids = [7] * 128 if i < 27 else [8] + [7] * 127
                pairs.append(self._pair(tmp, i, prefix_ids=ids))
            summary = qg.evaluate(pairs, "t16", "SKU", require_full=True)
            self.assertEqual(summary["first32_identical"], 27)
            self.assertEqual(summary["gate"]["verdict"], "fail")

    def test_not_measured_below_32_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [self._pair(tmp, i, prefix_ids=[7] * 128) for i in range(4)]
            summary = qg.evaluate(pairs, "t16", "SKU", require_full=True)
            self.assertEqual(summary["gate"]["verdict"], "not_measured")

    def test_tier_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [self._pair(tmp, 0, prefix_ids=[7] * 128, tier="t16")]
            with self.assertRaises(qg.GateError):
                qg.evaluate(pairs, "t32", "SKU")  # a T32 claim from a T16 run

    def test_compare_mode_refuses_mismatched_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.qlogits")
            b = os.path.join(tmp, "b.qlogits")
            lg = logits(8, peak=1)
            write_dump(a, prompt=[1], generated=[2], prefill_logits=lg, final_logits=lg)
            write_dump(b, prompt=[2], generated=[2], prefill_logits=lg, final_logits=lg)
            args = type("A", (), {
                "reference": [a], "candidate": [b], "tier": None,
                "device_tag": None, "require_full_prompt_count": False, "out": None,
            })
            with self.assertRaises(qg.GateError):
                qg.compare_mode(args)

    def test_long_generation_signals(self):
        self.assertEqual(qg._repeat_signal([1, 1, 2, 3])["longest_single_id_run"], 2)
        self.assertTrue(qg._degenerate([5] * 1024))
        self.assertTrue(qg._degenerate([5, 6] * 512))
        self.assertTrue(qg._degenerate([220, 17, 15, 16, 22] * 204 + [220] * 4))
        rng = random.Random(7)
        self.assertTrue(qg._degenerate([5 if rng.random() < 0.5 else 6
                                        for _ in range(1024)]))
        self.assertFalse(qg._degenerate(list(range(1024))))


class KvDtypeGateTests(unittest.TestCase):
    """The q3 gate (opt_plan/05 section 4): FP32-KV vs FP8-KV of ONE weight format."""

    def _pairs(self, tmp, n=16, kv_ref="f32", kv_cand="fp8", agreement=1.0,
               weight="bf16", weight_cand=None):
        pairs = []
        for i in range(n):
            prompt = [100 + i] * 8
            gen_ref = list(range(64))
            keep = int(round(len(gen_ref) * agreement))
            gen_cand = gen_ref[:keep] + [999] * (len(gen_ref) - keep)
            lg = logits(8, peak=1)
            ref = os.path.join(tmp, "r%d.qlogits" % i)
            cand = os.path.join(tmp, "c%d.qlogits" % i)
            write_dump(ref, prompt=prompt, generated=gen_ref, prefill_logits=lg,
                       final_logits=lg, fmt="%s+kv_%s" % (weight, kv_ref))
            write_dump(cand, prompt=prompt, generated=gen_cand, prefill_logits=lg,
                       final_logits=lg,
                       fmt="%s+kv_%s" % (weight_cand or weight, kv_cand))
            pairs.append((qg.Dump(ref), qg.Dump(cand)))
        return pairs

    @staticmethod
    def _needle(hits=True):
        return {
            "kind": "needle_long_context",
            "prompts": 4,
            "context_tokens_requested": 8192,
            "max_new_tokens": 16,
            "reference_hits": 4,
            "candidate_hits": 4 if hits else 3,
            "mean_ids_agreement": 1.0 if hits else 0.75,
            "verdict": "pass" if hits else "fail",
            "per_prompt": [{"prompt": 0, "prompt_tokens": 8190, "passkey": "1234567",
                            "reference_hit": True, "candidate_hit": hits,
                            "ids_agreement": 1.0 if hits else 0.75}],
        }

    def test_format_carries_both_axes(self):
        self.assertEqual(qg.parse_dump_format("bf16+kv_fp8_e4m3"),
                         ("bf16", "fp8_e4m3"))
        # A dump written before the KV axis existed reports UNKNOWN, never a guess.
        self.assertEqual(qg.parse_dump_format("w4"), ("w4", None))

    def test_pass_at_98_percent_with_needle(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = qg.evaluate(self._pairs(tmp), "t16", "SKU", require_full=True,
                                  gate="q3", needle=self._needle(),
                                  reference_kv_dtype="f32", candidate_kv_dtype="fp8")
            self.assertEqual(summary["gate"]["verdict"], "pass")
            self.assertEqual(summary["gate"]["needle_gate"], "pass")
            self.assertEqual(summary["kv_dtype"], {"reference": "f32",
                                                   "candidate": "fp8"})
            self.assertEqual(summary["weight_format"], "bf16")

    def test_fail_below_98_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = qg.evaluate(self._pairs(tmp, agreement=0.90), "t16", "SKU",
                                  require_full=True, gate="q3", needle=self._needle(),
                                  reference_kv_dtype="f32", candidate_kv_dtype="fp8")
            self.assertEqual(summary["gate"]["verdict"], "fail")

    def test_needle_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = qg.evaluate(self._pairs(tmp), "t16", "SKU", require_full=True,
                                  gate="q3", needle=None,
                                  reference_kv_dtype="f32", candidate_kv_dtype="fp8")
            self.assertEqual(summary["gate"]["needle_gate"], "missing")
            self.assertEqual(summary["gate"]["verdict"], "fail")

    def test_failed_needle_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = qg.evaluate(self._pairs(tmp), "t16", "SKU", require_full=True,
                                  gate="q3", needle=self._needle(hits=False),
                                  reference_kv_dtype="f32", candidate_kv_dtype="fp8")
            self.assertEqual(summary["gate"]["verdict"], "fail")

    def test_refuses_unlabelled_dumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = self._pairs(tmp)
            # Rewrite the reference dumps without the KV label: the gate must not
            # guess which dtype produced them.
            for i in range(len(pairs)):
                write_dump(os.path.join(tmp, "r%d.qlogits" % i),
                           prompt=[100 + i] * 8, generated=list(range(64)),
                           prefill_logits=logits(8, peak=1),
                           final_logits=logits(8, peak=1), fmt="bf16")
            pairs = [(qg.Dump(os.path.join(tmp, "r%d.qlogits" % i)),
                      qg.Dump(os.path.join(tmp, "c%d.qlogits" % i)))
                     for i in range(16)]
            with self.assertRaises(qg.GateError):
                qg.evaluate(pairs, "t16", "SKU", gate="q3", needle=self._needle())

    def test_refuses_same_kv_dtype_on_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(qg.GateError):
                qg.evaluate(self._pairs(tmp, kv_cand="f32"), "t16", "SKU", gate="q3",
                            needle=self._needle())

    def test_refuses_a_different_requested_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(qg.GateError):
                qg.evaluate(self._pairs(tmp), "t16", "SKU", gate="q3",
                            needle=self._needle(), reference_kv_dtype="bf16")

    def test_refuses_mixed_weight_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(qg.GateError):
                qg.evaluate(self._pairs(tmp, weight_cand="w4"), "t16", "SKU", gate="q3",
                            needle=self._needle(), reference_kv_dtype="f32",
                            candidate_kv_dtype="fp8")


class NeedleDocumentTests(unittest.TestCase):
    def test_document_contains_the_passkey_at_depth(self):
        text = qg.build_needle_document(1024, "1234567", 0.5)
        self.assertIn("The pass key is 1234567.", text)
        pos = text.index("The pass key is")
        self.assertGreater(pos, len(text) * 0.3)
        self.assertLess(pos, len(text) * 0.7)
        self.assertIn("What is the pass key?", text)

    def test_document_scales_with_the_context(self):
        small = qg.build_needle_document(1024, "1234567", 0.1)
        large = qg.build_needle_document(8192, "1234567", 0.1)
        self.assertGreater(len(large), len(small) * 4)

    def test_depth_is_validated(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(qg.GateError):
                qg.build_needle_document(128, "1234567", bad)


class PromptGeneratorTests(unittest.TestCase):
    def test_matches_the_benchmark_harness(self):
        for length in (1, 8, 128, 129, 1000):
            self.assertEqual(qg.synthetic_prompt_ids(length),
                             bq.synthetic_prompt_ids(length))

    def test_ids_stay_in_range(self):
        ids = qg.synthetic_prompt_ids(1024)
        self.assertEqual(len(ids), 1024)
        self.assertTrue(all(qg.PROMPT_ID_MIN <= i < qg.PROMPT_ID_MAX for i in ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
