#!/usr/bin/env python3
"""Stdlib unit tests for benchmark_qwen.py; no GPU, model or binary required.

Run like the other script tests (see scripts/NSIGHT.md):

    python scripts/test_benchmark_qwen.py
    python -m unittest discover -s scripts -p "test_*.py" -v

Nothing here launches example_qwen, nvidia-smi or a real benchmark: the
execution path is exercised by replacing benchmark_qwen.run_process with a
synthetic CLI log, and every nvidia-smi query is replaced by an injected
`capture` callable.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import benchmark_qwen as bq


# ---------------------------------------------------------------------------
# A synthetic example_qwen that satisfies every parser in the harness
# ---------------------------------------------------------------------------

def prompt_ids_from_cmd(cmd: list[str]) -> list[int]:
    """The arm's ids, whichever transport the command used."""
    if "--synthetic-prompt-len" in cmd:
        length = int(cmd[cmd.index("--synthetic-prompt-len") + 1])
        return bq.synthetic_prompt_ids(length)
    if "--tokens-file" in cmd:
        path = Path(cmd[cmd.index("--tokens-file") + 1])
        return [int(t) for t in path.read_text(encoding="utf-8").split()]
    tokens = cmd[cmd.index("--tokens") + 1]
    return [int(t) for t in tokens.split(",")]


def fake_example_qwen_stdout(cmd: list[str], *,
                             max_context_override: str | None = None) -> str:
    """One example_qwen log matching every line benchmark_qwen parses.

    Ids depend only on the prompt length and the decode budget, never on the
    engine version -- that is exactly the property --require-id-match exists to
    prove across versions.
    """
    ids_in = prompt_ids_from_cmd(cmd)
    prompt_len = len(ids_in)
    max_new = int(cmd[cmd.index("--max-new-tokens") + 1])
    max_context = (max_context_override
                   or cmd[cmd.index("--max-context") + 1])
    ids = [1000 + i * 7 + prompt_len for i in range(max_new)]
    steps = max_new - 1
    return "\n".join([
        f"prompt: {prompt_len} tokens",
        "prompt ids:" + ("" if prompt_len > 64
                         else ''.join(' ' + str(t) for t in ids_in)),
        "engine: kernel=dsl storage=fp8 max_context=%s prefill_chunk=64 "
        "gdn=wy-ut-64 cache=12.50 MiB workspace=3.25 MiB shaders=321 "
        "(compile 4.5 ms)" % max_context,
        # Phase 3 provenance: prefill kernel + workspace split + the chunking the
        # engine actually selected (opt_plan/02 sections 3 and 4).
        "prefill gemm: mt(mt=16)",
        "workspace: total=203.23 MiB chunk=202.25 MiB pool_per_row=44096 floats",
        "prefill chunking: mode=adaptive configured_chunk=64 "
        "configured_max=1024 backend_limit=1024 selected=64 chunks=%d"
        % max(1, (prompt_len + 63) // 64),
        "generated ids (%d):%s" % (len(ids), "".join(" %d" % i for i in ids)),
        "stop: max-new-tokens budget",
        "timing: host-load 1.0 ms | weight-upload 2.0 ms | engine-compile "
        "3.0 ms | TTFT(prefill %d tokens) 10.0 ms | decode 5.00 ms/token "
        "(%d steps)" % (prompt_len, steps),
        "engine final: position=%d shaders=321 cuda-graph-steps=%d updates=0"
        % (prompt_len + steps, steps),
    ]) + "\n"


def fake_run_process(cmd, log_base, *, timeout, label):
    """Stand-in for benchmark_qwen.run_process (no process is started)."""
    return {
        "command": cmd,
        "cwd": str(bq.REPO_ROOT),
        "returncode": 0,
        "process_wall_s": 1.5,
        "stdout_log": str(log_base) + ".stdout.log",
        "stderr_log": str(log_base) + ".stderr.log",
        "stdout": fake_example_qwen_stdout(cmd),
        "stderr": "",
    }


def make_inputs(root: Path) -> tuple[Path, Path]:
    """Fake exe + model files: the harness only stats/opens paths, and the
    process itself is mocked, so a few bytes are enough."""
    exe = root / "example_qwen.exe"
    exe.write_bytes(b"MZ")
    model = root / "Qwen3.5-4B-BF16.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 32)
    return exe, model


def quiet_main(argv: list[str]) -> tuple[int, str, str]:
    """Run bq.main() with its stdout/stderr captured."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = bq.main(argv)
    return code, out.getvalue(), err.getvalue()


# What benchmark_qwen records on a machine without nvidia-smi.
NO_GPU_IDENTITY = {
    "nvidia_smi_available": False,
    "gpu_index": 0,
    "name": None,
    "driver_version": None,
    "memory_total_mib": None,
    "compute_capability": None,
}


@contextlib.contextmanager
def no_gpu():
    """Make every nvidia-smi query report "unavailable" (no process starts)."""
    with mock.patch.object(bq, "query_gpu_identity",
                           return_value=dict(NO_GPU_IDENTITY)), \
            mock.patch.object(bq, "query_gpu_memory", return_value=None), \
            mock.patch.object(bq, "query_gpu_used_mib", return_value=None), \
            mock.patch.object(bq, "query_gpu_state", return_value=None):
        yield


# ---------------------------------------------------------------------------
# Feature 1: deterministic synthetic prompts
# ---------------------------------------------------------------------------

class PromptGeneratorTests(unittest.TestCase):
    def test_same_length_gives_identical_ids_twice(self) -> None:
        first = bq.synthetic_prompt_ids(512)
        second = bq.synthetic_prompt_ids(512)
        self.assertEqual(first, second)

    def test_length_is_exact(self) -> None:
        for length in (1, 2, 128, 512, 2048):
            self.assertEqual(len(bq.synthetic_prompt_ids(length)), length)

    def test_ids_are_inside_the_vocabulary_and_the_documented_range(self) -> None:
        ids = bq.synthetic_prompt_ids(2048)
        self.assertTrue(all(bq.PROMPT_ID_MIN <= i < bq.PROMPT_ID_MAX for i in ids))
        # The whole documented range must be valid for the Qwen3.5-4B vocabulary
        # the CLI checks, otherwise a sweep would be rejected at prefill.
        self.assertLess(bq.PROMPT_ID_MAX, bq.QWEN_VOCAB_SIZE)
        self.assertTrue(all(0 <= i < bq.QWEN_VOCAB_SIZE for i in ids))
        self.assertGreater(len(set(ids)), 2000 - 100)

    def test_different_lengths_produce_different_prompts(self) -> None:
        short = bq.synthetic_prompt_ids(128)
        long = bq.synthetic_prompt_ids(512)
        self.assertNotEqual(short, long[:128])

    def test_generator_rejects_empty_prompt(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "length must be >= 1"):
            bq.synthetic_prompt_ids(0)

    def test_ids_to_arg_is_the_cli_spelling(self) -> None:
        self.assertEqual(bq.ids_to_arg([1, 2, 3]), "1,2,3")
        self.assertEqual(len(bq.ids_to_arg(bq.synthetic_prompt_ids(64))
                             .split(",")), 64)


# ---------------------------------------------------------------------------
# Feature 2: prompt/context sweeps + arm attribution
# ---------------------------------------------------------------------------

class ArmPlanningTests(unittest.TestCase):
    def args(self, *argv: str):
        return bq.parse_args(["--max-new-tokens", "8", *argv])

    def test_legacy_single_arm_from_tokens(self) -> None:
        args = self.args("--tokens", "1,2,3", "--max-context", "64")
        arms, max_context, sweeping = bq.plan_sweep(args)
        self.assertFalse(sweeping)
        self.assertEqual(len(arms), 1)
        self.assertEqual(arms[0]["prompt_length"], 3)
        self.assertEqual(arms[0]["tokens"], "1,2,3")
        self.assertEqual(max_context, 64)

    def test_legacy_prompt_must_fit_max_context(self) -> None:
        args = self.args("--tokens", ",".join("1" for _ in range(100)),
                         "--max-context", "64")
        with self.assertRaisesRegex(bq.BenchmarkError, "exceeds max-context"):
            bq.plan_sweep(args)

    def test_prompt_lengths_drive_tokens_not_text(self) -> None:
        args = self.args("--prompt-lengths", "128,512")
        arms, max_context, sweeping = bq.plan_sweep(args)
        self.assertTrue(sweeping)
        self.assertEqual([a["prompt_length"] for a in arms], [128, 512])
        for arm in arms:
            self.assertEqual(arm["tokens"],
                             bq.ids_to_arg(bq.synthetic_prompt_ids(
                                 arm["prompt_length"])))
            self.assertEqual(len(arm["tokens"].split(",")), arm["prompt_length"])
        self.assertEqual([a["key"] for a in arms], ["P128-C520", "P512-C520"])
        # capacity must cover the longest prompt plus the decode budget
        self.assertEqual(max_context, 520)

    def test_context_lengths_set_the_prompt_and_the_bucket(self) -> None:
        args = self.args("--context-lengths", "128,512,2048")
        arms, max_context, sweeping = bq.plan_sweep(args)
        self.assertTrue(sweeping)
        self.assertEqual([a["context_length"] for a in arms], [128, 512, 2048])
        # the prompt length equals C so the decode loop runs at ctx ~= C
        self.assertEqual([a["prompt_length"] for a in arms], [128, 512, 2048])
        self.assertEqual(max_context, 2056)

    def test_both_flags_are_a_cross_product_and_reject_overflow(self) -> None:
        args = self.args("--prompt-lengths", "128,512", "--context-lengths", "4096")
        arms, max_context, _ = bq.plan_sweep(args)
        self.assertEqual([a["key"] for a in arms],
                         ["P128-C4096", "P512-C4096"])
        self.assertEqual(max_context, 4096)
        bad = self.args("--prompt-lengths", "512", "--context-lengths", "128")
        with self.assertRaisesRegex(bq.BenchmarkError,
                                    "does not fit in the requested context"):
            bq.plan_sweep(bad)

    def test_length_lists_are_validated(self) -> None:
        for text, message in (("128,,512", "empty entry"), ("abc", "positive"),
                              ("0", ">= 1"), ("128,128", "duplicate")):
            with self.assertRaisesRegex(bq.BenchmarkError, message):
                bq.parse_length_list(text, "--context-lengths")

    def test_repeat_rules_never_allow_a_single_run(self) -> None:
        sweeping = self.args("--prompt-lengths", "128")
        plain = self.args()
        self.assertEqual(bq.resolve_repeat(plain, False), bq.DEFAULT_REPEAT)
        self.assertEqual(bq.resolve_repeat(sweeping, True),
                         bq.SWEEP_REPEAT_DEFAULT)
        explicit = self.args("--repeat", "5")
        self.assertEqual(bq.resolve_repeat(explicit, True), 5)
        with self.assertRaisesRegex(bq.BenchmarkError, "single run"):
            bq.resolve_repeat(self.args("--repeat", "1"), False)
        with self.assertRaisesRegex(bq.BenchmarkError, "for a sweep"):
            bq.resolve_repeat(self.args("--repeat", "3"), True)

    def test_source_quant_flag_overrides_the_file_name_inference(self) -> None:
        self.assertEqual(bq.resolve_source_quant("q8_0", "x-BF16.gguf")[0],
                         "q8_0")
        self.assertEqual(bq.resolve_source_quant("q8_0", "x-BF16.gguf")[1],
                         "flag")
        self.assertEqual(bq.resolve_source_quant(None, "Qwen-Q8_0.gguf"),
                         ("q8_0", "inferred-from-model-filename"))
        self.assertEqual(bq.resolve_source_quant(None, "Qwen-BF16.gguf"),
                         ("bf16", "inferred-from-model-filename"))
        self.assertEqual(bq.resolve_source_quant(None, "mystery.gguf"),
                         ("unspecified", "unspecified"))

    def test_long_prompt_command_line_is_refused_before_launch(self) -> None:
        tokens = bq.ids_to_arg(bq.synthetic_prompt_ids(8192))
        cmd = ["example_qwen.exe", "--tokens", tokens]
        message = bq.command_length_error(cmd, max_chars=30000, label="arm")
        self.assertIsNotNone(message)
        self.assertIn("32767", message)
        self.assertIn("Nothing was launched", message)
        short = ["example_qwen.exe", "--tokens", "1,2,3"]
        self.assertIsNone(bq.command_length_error(short, max_chars=30000,
                                                  label="arm"))


# ---------------------------------------------------------------------------
# Provenance parsing (engine line, malformed output)
# ---------------------------------------------------------------------------

class ParsingTests(unittest.TestCase):
    def test_engine_line_is_parsed_for_provenance(self) -> None:
        cmd = ["exe", "--tokens", "1,2,3", "--max-new-tokens", "3",
               "--max-context", "64"]
        engine = bq.parse_engine_lines(fake_example_qwen_stdout(cmd), "label")
        self.assertEqual(engine["kernel"], "dsl")
        self.assertEqual(engine["compiled_shaders"], 321)
        self.assertEqual(engine["cache_mib"], 12.5)
        self.assertEqual(engine["reported_max_context"], 64)
        self.assertEqual(engine["cuda_graph_steps"], 2)
        self.assertEqual(engine["cuda_graph_updates"], 0)
        self.assertEqual(engine["engine_compile_ms_cli"], 4.5)
        # Phase 3 provenance is parsed and range-checked, never guessed.
        self.assertEqual(engine["prefill_gemm"], "mt(mt=16)")
        self.assertEqual(engine["prefill_gemm_mt"], 16)
        self.assertEqual(engine["workspace_total_mib"], 203.23)
        self.assertEqual(engine["workspace_chunk_mib"], 202.25)
        self.assertEqual(engine["workspace_pool_floats_per_row"], 44096)
        self.assertEqual(engine["prefill_chunking"],
                         {"mode": "adaptive", "configured_chunk": 64,
                          "configured_max": 1024, "backend_limit": 1024,
                          "selected": 64, "chunks": 1})

    def _chunking_log(self, line: str) -> str:
        """One minimal log carrying the phase 3 chunking/gemm lines."""
        return "\n".join([
            "engine: kernel=dsl storage=fp8 max_context=64 prefill_chunk=64 "
            "gdn=wy-ut-64 cache=12.50 MiB workspace=3.25 MiB shaders=321 "
            "(compile 4.5 ms)",
            line,
            "generated ids (1): 5",
            "engine final: position=2 shaders=321 cuda-graph-steps=0 updates=0",
        ]) + "\n"

    def test_prefill_chunking_provenance_rules(self) -> None:
        ok = ("prefill chunking: mode=adaptive configured_chunk=1024 "
              "configured_max=1024 backend_limit=1024 selected=256 chunks=4")
        info = bq.parse_engine_lines(self._chunking_log(ok), "label")[
            "prefill_chunking"]
        self.assertEqual(info["selected"], 256)
        self.assertEqual(info["chunks"], 4)
        # A fixed-chunk run must report the chunk it was configured with.
        fixed = ("prefill chunking: mode=fixed configured_chunk=64 "
                 "configured_max=1024 backend_limit=1024 selected=64 chunks=16")
        self.assertEqual(
            bq.parse_engine_lines(self._chunking_log(fixed), "label")[
                "prefill_chunking"]["selected"], 64)
        # selected > backend limit, selected > configured, fixed-mode drift and
        # a zero chunk count all abort instead of producing a wrong summary.
        for bad in (
            "prefill chunking: mode=adaptive configured_chunk=1024 "
            "configured_max=1024 backend_limit=128 selected=256 chunks=4",
            "prefill chunking: mode=adaptive configured_chunk=64 "
            "configured_max=64 backend_limit=1024 selected=256 chunks=4",
            "prefill chunking: mode=fixed configured_chunk=64 "
            "configured_max=1024 backend_limit=1024 selected=256 chunks=4",
            "prefill chunking: mode=adaptive configured_chunk=1024 "
            "configured_max=1024 backend_limit=1024 selected=256 chunks=0",
        ):
            with self.assertRaises(bq.BenchmarkError):
                bq.parse_engine_lines(self._chunking_log(bad), "label")
        dup = self._chunking_log(ok).replace(
            "generated ids (1): 5", ok + "\ngenerated ids (1): 5")
        with self.assertRaisesRegex(bq.BenchmarkError, "prefill chunking"):
            bq.parse_engine_lines(dup, "label")

    def test_engine_line_with_prefill_attn_field_parses(self) -> None:
        """The engine line carries prefill_attn=... since Phase 2; it must parse.

        The field was added to example_qwen's `engine:` line without a parser
        update, so every REAL run failed to parse while the synthetic fixtures
        passed. This pins the real line shape (taken verbatim from a Phase-5 run).
        """
        real = ("engine: kernel=dsl storage=bf16_packed max_context=4096 "
                "prefill_chunk=1024 gdn=wy-ut-64 prefill_attn=tiled(q2k2,4096B) "
                "cache=306.25 MiB workspace=203.86 MiB shaders=35 (compile 247.7 ms)")
        txt = "\n".join([
            real,
            "generated ids (1): 5",
            "engine final: position=2 shaders=35 cuda-graph-steps=1 updates=0",
        ]) + "\n"
        engine = bq.parse_engine_lines(txt, "label")
        self.assertEqual(engine["kernel"], "dsl")
        self.assertEqual(engine["weight_storage"], "bf16_packed")
        self.assertEqual(engine["reported_max_context"], 4096)
        self.assertEqual(engine["cache_mib"], 306.25)
        self.assertEqual(engine["workspace_mib"], 203.86)
        self.assertEqual(engine["compiled_shaders"], 35)

    def test_decode_attn_provenance(self) -> None:
        """Phase 5: the split-KV decode attention line is parsed and validated."""
        def log(line: str) -> str:
            return "\n".join([
                "engine: kernel=dsl storage=bf16_packed max_context=4096 "
                "prefill_chunk=1024 gdn=wy-ut-64 cache=306.25 MiB workspace=203.49 MiB "
                "shaders=39 (compile 8670.3 ms)",
                line,
                "generated ids (1): 5",
                "engine final: position=2 shaders=39 cuda-graph-steps=1 updates=0",
            ]) + "\n"

        split = ("decode attn: requested=split active=split threshold=512 "
                 "max_splits=40 k_tile=8 shared=16384 partial_bytes=659456 "
                 "blocks=176 splits=4 kv_len=2048 dedup=1 sm=0 sm_source=unqueryable")
        info = bq.parse_engine_lines(log(split), "label")["decode_attn"]
        self.assertEqual(info["requested"], "split")
        self.assertEqual(info["active"], "split")
        self.assertEqual(info["threshold"], 512)
        self.assertEqual(info["max_splits"], 40)
        self.assertEqual(info["k_tile"], 8)
        self.assertEqual(info["blocks"], 176)
        self.assertEqual(info["splits"], 4)
        self.assertEqual(info["kv_len"], 2048)
        self.assertTrue(info["dedup"])
        # A backend that answers no SM-count query (dx/vk) reports 0: 0 must survive
        # as 0 so the report can say "blocks/SM unavailable" instead of dividing by
        # a guess.
        self.assertEqual(info["sm_count"], 0)
        self.assertEqual(info["sm_source"], "unqueryable")
        # ... and a backend that DOES answer keeps the real count and its exact
        # source label (cuda, see patches/luisa-cuda-multiprocessor-query.patch):
        # 176 blocks / 80 SMs is the gate-11 blocks/SM evidence.
        cuda_split = ("decode attn: requested=split active=split threshold=512 "
                      "max_splits=40 k_tile=8 shared=16384 partial_bytes=660480 "
                      "blocks=176 splits=4 kv_len=2048 dedup=1 sm=80 "
                      "sm_source=device_query:multi_processor_count")
        cuda_info = bq.parse_engine_lines(log(cuda_split), "label")["decode_attn"]
        self.assertEqual(cuda_info["sm_count"], 80)
        self.assertEqual(cuda_info["sm_source"],
                         "device_query:multi_processor_count")
        # A legacy run reports no threshold and no dedup.
        legacy = ("decode attn: requested=legacy active=legacy threshold=0 "
                  "max_splits=0 k_tile=0 shared=0 partial_bytes=0 "
                  "blocks=16 splits=0 kv_len=2048 dedup=0 sm=0 sm_source=unqueryable")
        legacy_info = bq.parse_engine_lines(log(legacy), "label")["decode_attn"]
        self.assertFalse(legacy_info["dedup"])
        self.assertEqual(legacy_info["active"], "legacy")
        # Fail-closed: a claim that contradicts itself aborts instead of producing a
        # summary whose roof model cannot be trusted.
        for bad in (
            # dedup=1 below the threshold it reports
            "decode attn: requested=split active=split threshold=512 "
            "max_splits=40 k_tile=8 shared=16384 partial_bytes=659456 "
            "blocks=176 splits=1 kv_len=128 dedup=1 sm=0 sm_source=unqueryable",
            # splits above the compile-time grid bound
            "decode attn: requested=split active=split threshold=512 "
            "max_splits=8 k_tile=8 shared=16384 partial_bytes=659456 "
            "blocks=176 splits=16 kv_len=8192 dedup=1 sm=0 sm_source=unqueryable",
            # dedup claimed by a legacy engine
            "decode attn: requested=legacy active=legacy threshold=0 "
            "max_splits=0 k_tile=0 shared=0 partial_bytes=0 "
            "blocks=16 splits=0 kv_len=2048 dedup=1 sm=0 sm_source=unqueryable",
        ):
            with self.assertRaises(bq.BenchmarkError):
                bq.parse_engine_lines(log(bad), "label")
        # Logs from earlier phases have no such line: the field is None and the roof
        # model keeps the legacy 4x term.
        old = bq.parse_engine_lines(
            "\n".join([
                "engine: kernel=dsl storage=fp8 max_context=64 prefill_chunk=64 "
                "gdn=wy-ut-64 cache=12.50 MiB workspace=3.25 MiB shaders=321 "
                "(compile 4.5 ms)",
                "generated ids (1): 5",
                "engine final: position=2 shaders=321 cuda-graph-steps=0 updates=0",
            ]) + "\n", "label")
        self.assertIsNone(old["decode_attn"])
        self.assertFalse(bq.decode_attn_dedup_applies(old, 8192))

    def test_prefill_gemm_provenance(self) -> None:
        legacy = self._chunking_log("prefill gemm: legacy")
        engine = bq.parse_engine_lines(legacy, "label")
        self.assertEqual(engine["prefill_gemm"], "legacy")
        self.assertEqual(engine["prefill_gemm_mt"], 0)
        # Logs without the phase 3 lines still parse (earlier phases' artifacts).
        old_log = "\n".join([
            "engine: kernel=dsl storage=fp8 max_context=64 prefill_chunk=64 "
            "gdn=wy-ut-64 cache=12.50 MiB workspace=3.25 MiB shaders=321 "
            "(compile 4.5 ms)",
            "generated ids (1): 5",
            "engine final: position=2 shaders=321 cuda-graph-steps=0 updates=0",
        ]) + "\n"
        engine = bq.parse_engine_lines(old_log, "label")
        self.assertIsNone(engine["prefill_gemm"])
        self.assertIsNone(engine["prefill_chunking"])


    def test_missing_engine_line_aborts(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "engine"):
            bq.parse_engine_lines("generated ids (1): 5\n", "label")

    def test_malformed_run_output_still_aborts(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "generated ids"):
            bq.parse_run_output("nonsense\n", version="dsl-fp8", prompt_len=3,
                                max_new_tokens=2, label="label")


# ---------------------------------------------------------------------------
# Feature 3: VRAM sampling, including the no-nvidia-smi path
# ---------------------------------------------------------------------------

def capture_returning(text: str | None):
    def capture(cmd, *, timeout: float = 20.0) -> str | None:
        return text
    return capture


class VramTests(unittest.TestCase):
    def test_no_nvidia_smi_records_null_and_keeps_working(self) -> None:
        unavailable = capture_returning(None)
        self.assertIsNone(bq.query_gpu_memory(capture=unavailable))
        self.assertIsNone(bq.query_gpu_used_mib(capture=unavailable))
        identity = bq.query_gpu_identity(capture=unavailable)
        self.assertFalse(identity["nvidia_smi_available"])
        self.assertIsNone(identity["name"])
        self.assertIsNone(identity["compute_capability"])
        poller = bq.VramPoller(0.02, capture=unavailable)
        poller.start()
        self.assertEqual(poller.stop(), [])
        self.assertFalse(poller.available)

    def test_missing_nvidia_smi_executable_does_not_raise(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("gone")):
            self.assertIsNone(bq.query_gpu_memory())
            self.assertIsNone(bq.query_gpu_identity()["name"])

    def test_poller_collects_coarse_samples(self) -> None:
        poller = bq.VramPoller(0.02, capture=capture_returning("0, 4242"))
        poller.start()
        samples = poller.stop()
        self.assertTrue(poller.available)
        self.assertGreaterEqual(len(samples), 1)
        self.assertTrue(all(s["used_mib"] == 4242 for s in samples))
        self.assertTrue(all(s["t_s"] >= 0 for s in samples))

    def test_summary_maps_samples_onto_the_three_checkpoints(self) -> None:
        summary = bq.summarize_vram_samples(
            before_start_mib=100,
            during_samples=[{"t_s": 0.2, "used_mib": 900},
                            {"t_s": 0.4, "used_mib": 1500}],
            after_exit_mib=50, interval_s=0.2)
        self.assertEqual(summary["before_start_mib"], 100)
        self.assertEqual(summary["after_load_prefill_mib"], 1500)
        self.assertEqual(summary["after_decode_mib"], 1500)
        self.assertEqual(summary["after_exit_mib"], 50)
        self.assertEqual(summary["peak_used_mib"], 1500)
        self.assertIn("COARSE", summary["labels"]["after_load_prefill_mib"])

    def test_empty_samples_are_null_not_zero(self) -> None:
        summary = bq.summarize_vram_samples(
            before_start_mib=None, during_samples=[], after_exit_mib=None,
            interval_s=0.2)
        for key in ("before_start_mib", "after_load_prefill_mib",
                    "after_decode_mib", "after_exit_mib", "peak_used_mib"):
            self.assertIsNone(summary[key])

    def test_preflight_decision(self) -> None:
        gpu = {"gpu_index": 0, "name": "RTX 4060"}
        low = bq.preflight_vram(64, min_free_mib=512, gpu=gpu)
        self.assertEqual(low["status"], "low")
        self.assertIn("ABORTING before launch", low["message"])
        self.assertIn("RTX 4060", low["message"])
        self.assertEqual(bq.preflight_vram(4096, min_free_mib=512,
                                           gpu=gpu)["status"], "ok")
        unavailable = bq.preflight_vram(None, min_free_mib=512, gpu=gpu)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["free_mib"])


# ---------------------------------------------------------------------------
# Feature 5: id reference round trip
# ---------------------------------------------------------------------------

def sample_arms() -> list[dict]:
    return [
        {"arm": "P128-C4096", "version": "dsl-fp8", "prompt_length": 128,
         "context_length": 4096, "generated_ids": [11, 22, 33],
         "stop": "max-new-tokens budget"},
        {"arm": "P512-C4096", "version": "dsl-fp8", "prompt_length": 512,
         "context_length": 4096, "generated_ids": [44, 55],
         "stop": "EOS (<|im_end|>)"},
    ]


class ReferenceTests(unittest.TestCase):
    def build(self) -> dict:
        return bq.build_id_reference(
            backend="cuda", model="data/Qwen3.5-4B-BF16.gguf",
            source_quant="bf16", max_new_tokens=8, arms=sample_arms())

    def test_round_trip_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.json"
            bq.write_id_reference(path, self.build())
            loaded = bq.load_id_reference(path)
            doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], bq.ID_REFERENCE_SCHEMA)
        self.assertEqual(doc["version"], bq.ID_REFERENCE_VERSION)
        self.assertEqual(doc["backend"], "cuda")
        self.assertEqual(doc["source_quant"], "bf16")
        self.assertEqual(sorted(doc["arms"]), ["P128-C4096", "P512-C4096"])
        self.assertEqual(doc["arms"]["P128-C4096"]["prompt_length"], 128)
        self.assertEqual(doc["arms"]["P128-C4096"]["tokens"], 8)
        self.assertEqual(doc["arms"]["P128-C4096"]["ids"], [11, 22, 33])
        result = bq.compare_ids_to_reference(loaded, arm="P128-C4096",
                                            ids=[11, 22, 33])
        self.assertTrue(result["match"])
        self.assertIsNone(result["first_difference_index"])

    def test_mismatch_names_the_arm_and_the_first_differing_index(self) -> None:
        reference = self.build()
        with self.assertRaisesRegex(bq.BenchmarkError,
                                    "P128-C4096.*index 1") as ctx:
            bq.compare_ids_to_reference(reference, arm="P128-C4096",
                                        ids=[11, 99, 33])
        self.assertIn("reference 22 vs run 99", str(ctx.exception))

    def test_shorter_run_is_also_a_mismatch(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "index 2"):
            bq.compare_ids_to_reference(self.build(), arm="P128-C4096",
                                        ids=[11, 22])

    def test_unknown_arm_fails_closed(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "not present"):
            bq.compare_ids_to_reference(self.build(), arm="P2048-C4096",
                                        ids=[1])

    def test_broken_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_json = root / "bad.json"
            bad_json.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(bq.BenchmarkError, "not JSON"):
                bq.load_id_reference(bad_json)
            wrong = root / "wrong.json"
            wrong.write_text(json.dumps({"version": 99, "arms": {}}),
                             encoding="utf-8")
            with self.assertRaisesRegex(bq.BenchmarkError, "version"):
                bq.load_id_reference(wrong)
            empty = root / "empty.json"
            empty.write_text(json.dumps({"version": 1, "arms": {}}),
                             encoding="utf-8")
            with self.assertRaisesRegex(bq.BenchmarkError, "no 'arms'"):
                bq.load_id_reference(empty)
            with self.assertRaisesRegex(bq.BenchmarkError, "cannot read"):
                bq.load_id_reference(root / "missing.json")


# ---------------------------------------------------------------------------
# Feature 6: markdown report from collected JSON
# ---------------------------------------------------------------------------

def synthetic_summary() -> dict:
    """A summary shaped exactly like the harness writes it (no run needed)."""
    return {
        "run_id": "run_20260101_000000",
        "run_dir": "C:/tmp/run_20260101_000000",
        "tier": "t16",
        "device_tag": "RTX_4080_SUPER_16G",
        "bandwidth_gbps": None,
        "roof_model": bq.ROOF_MODEL_NOTE,
        "context": {
            "started_utc": "2026-01-01T00:00:00+00:00",
            "backend": "cuda",
            "model": "C:/repo/data/Qwen3.5-4B-BF16.gguf",
            "source_quant": "bf16",
            "source_quant_origin": "flag",
            "ttft_kind": "cold",
            "max_new_tokens": 32,
            "repeat": 5,
        },
        "arms": [{
            "arm": "P128-C4096", "version": "dsl-fp8",
            "prompt_length": 128, "context_length": 4096,
            "max_context": 4096, "ttft_kind": "cold", "source_quant": "bf16",
            "backend": "cuda", "model": "C:/repo/data/Qwen3.5-4B-BF16.gguf",
            "repeat": 5, "ids_match": True,
            "generated_ids": [1, 2, 3], "stop": "EOS (<|im_end|>)",
            "stats": {"new": {
                "n": 5,
                "ttft_ms": {"median": 41.5, "min": 40.0, "p95": 44.9,
                            "max": 45.0},
                "decode_ms_per_token": {"median": 8.25, "min": 8.0,
                                        "p95": 8.4, "max": 8.5},
                "decode_tokens_per_s": {"median": 121.21, "min": 117.65,
                                        "p95": 125.0, "max": 125.0},
                "host_load_ms": None, "weight_upload_ms": None,
                "engine_compile_ms": None,
            }},
            "vram": {"peak_used_mib": 6123, "sample_interval_s": 0.2,
                     "sample_count": 9, "before_start_mib": 700,
                     "after_load_prefill_mib": 6123, "after_decode_mib": 6100,
                     "after_exit_mib": 650},
            "engine": {"kernel": "dsl", "weight_storage": "fp8",
                       "reported_max_context": 4096, "reported_prefill_chunk": 64,
                       "gdn": "wy-ut-64", "cache_mib": 12.5,
                       "workspace_mib": 3.25, "compiled_shaders": 321,
                       "engine_compile_ms_cli": 4.5, "final_position": 159,
                       "cuda_graph_steps": 31, "cuda_graph_updates": 0},
        }],
    }


class ReportTests(unittest.TestCase):
    def test_required_columns_are_present(self) -> None:
        text = bq.render_report_md(synthetic_summary())
        header = next(line for line in text.splitlines()
                      if line.startswith("| version / arm"))
        for column in ("version / arm", "backend", "model", "quant source",
                       "max_context", "prompt length", "TTFT ms", "ms/token",
                       "tokens/s", "peak VRAM MiB", "repeats", "ids match"):
            self.assertIn(column, header)
        self.assertEqual(header.count("|"), len(bq.REPORT_COLUMNS) + 1)

    def test_numbers_come_from_the_summary(self) -> None:
        text = bq.render_report_md(synthetic_summary())
        row = next(line for line in text.splitlines()
                   if "P128-C4096" in line)
        self.assertIn("41.5", row)      # TTFT median
        self.assertIn("8.250", row)     # ms/token
        self.assertIn("121.21", row)    # tokens/s
        self.assertIn("6123", row)      # peak VRAM
        self.assertIn("bf16", row)      # source_quant
        self.assertIn("yes", row)       # ids match

    def test_missing_values_render_as_not_available(self) -> None:
        summary = synthetic_summary()
        summary["arms"][0]["vram"] = None
        summary["arms"][0]["ids_match"] = None
        row = next(line for line in bq.render_report_md(summary).splitlines()
                   if "P128-C4096" in line)
        self.assertIn("n/a", row)
        self.assertIn("not-checked", row)

    def test_empty_arms_do_not_crash(self) -> None:
        summary = synthetic_summary()
        summary["arms"] = []
        self.assertIn("no arms", bq.render_report_md(summary))


# ---------------------------------------------------------------------------
# Workflow: --help, --dry-run and a synthetic end-to-end run
# ---------------------------------------------------------------------------

class WorkflowTests(unittest.TestCase):
    def test_help_works_without_gpu_or_inputs(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            bq.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        for flag in ("--prompt-lengths", "--context-lengths", "--vram",
                     "--ttft-kind", "--require-id-match",
                     "--write-id-reference", "--report-md", "--source-quant"):
            self.assertIn(flag, text)

    # ------------------------------------------------------------------
    # Native Q8_0 (phase 8a / opt_plan 05 Q1): the dsl-q8 arm maps to the Q8_0
    # checkpoint on its own, and its storage class is modelled as an 8-bit
    # format everywhere the report reads it.
    # ------------------------------------------------------------------
    def test_q8_checkpoint_version_mismatch_is_refused(self) -> None:
        # A Q8_0 checkpoint with a non-q8 version, and one --model shared by a
        # mixed dsl-q8/other group, are both refused before any launch: the CLI
        # fails closed on them, so the harness must not schedule the arms.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q8_model = root / "model-Q8_0.gguf"
            q8_model.write_bytes(b"not a real gguf")
            with no_gpu():
                code, _, err = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--versions", "dsl-bf16", "--model", str(q8_model)])
                self.assertNotEqual(code, 0)
                self.assertIn("looks like a Q8_0 checkpoint", err)
                code, _, err = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--versions", "dsl-q8,dsl-fp8", "--model", str(q8_model)])
                self.assertNotEqual(code, 0)
                self.assertIn("--versions mixes dsl-q8", err)
                # dsl-q8 alone with an explicit Q8_0 path is accepted.
                code, out, _ = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--versions", "dsl-q8", "--model", str(q8_model)])
                self.assertEqual(code, 0)
                self.assertIn("--model " + str(q8_model), out)

    def test_dsl_q8_defaults_to_the_q8_0_checkpoint(self) -> None:
        self.assertEqual(bq.resolve_arm_model("dsl-q8", None),
                         "data/Qwen3.5-4B-Q8_0.gguf")
        self.assertEqual(bq.resolve_arm_model("dsl-bf16", None),
                         "data/Qwen3.5-4B-BF16.gguf")
        self.assertEqual(bq.resolve_arm_model("dsl-fp8", None),
                         "data/Qwen3.5-4B-BF16.gguf")
        # An explicit --model always wins (the CLI is what fails closed on a
        # mismatched checkpoint, never the harness).
        self.assertEqual(bq.resolve_arm_model("dsl-q8", "other.gguf"), "other.gguf")
        # The version -> format and version -> weight tables already carry the
        # q8 class (05 Q1 seam); keep them consistent with the new default.
        self.assertEqual(bq.VERSION_WEIGHT_FORMAT["dsl-q8"], "q8_0")
        self.assertIn("q8_0", bq.FORMAT_WEIGHT_GIB)
        # The engine reports `storage=q8_0_packed`; the roof byte model must know
        # that exact key (an unknown key yields n/a, never a guess).
        self.assertIn("q8_0_packed", bq.DOCUMENTED_WEIGHT_BYTES)
        self.assertAlmostEqual(bq.DOCUMENTED_WEIGHT_BYTES["q8_0_packed"],
                               bq.DOCUMENTED_WEIGHT_BYTES["q8_0"])
        # FP32 scale widening is real device traffic. Do not regress to the GGUF
        # source's 1.0625 B/weight total and omit those scales from the roof model.
        self.assertGreater(bq.DOCUMENTED_WEIGHT_BYTES["q8_0_packed"],
                           bq.DOCUMENTED_WEIGHT_BYTES["fp8_packed"])
        self.assertAlmostEqual(bq.DOCUMENTED_WEIGHT_BYTES["q8_0_packed"],
                               4.73424e9, delta=1e5)

    def test_dsl_q8_dry_run_passes_the_q8_0_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with no_gpu():
                code, out, _ = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--versions", "dsl-q8", "--output-dir", str(root / "never"),
                    "--max-context", "512"])
        self.assertEqual(code, 0)
        self.assertIn("--version dsl-q8", out)
        self.assertIn("--model data/Qwen3.5-4B-Q8_0.gguf", out)
        self.assertNotIn("--model data/Qwen3.5-4B-BF16.gguf", out)
        # 05 Q1's capacity class: the 8-bit column, i.e. 128 K is reachable on T16.
        self.assertIn("source_quant: q8_0", out)

    def test_dsl_q8_and_bf16_arms_use_their_own_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with no_gpu():
                code, out, _ = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--versions", "dsl-bf16,dsl-q8",
                    "--output-dir", str(root / "never"), "--max-context", "512"])
        self.assertEqual(code, 0)
        self.assertIn("--version dsl-bf16 --model data/Qwen3.5-4B-BF16.gguf", out)
        self.assertIn("--version dsl-q8 --model data/Qwen3.5-4B-Q8_0.gguf", out)

    def test_dry_run_needs_no_gpu_model_or_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "never-created"
            with no_gpu():
                code, out, _ = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--model", str(root / "missing.gguf"),
                    "--output-dir", str(out_dir), "--prompt-lengths", "128,512",
                    "--repeat", "5", "--vram", "--ttft-kind", "warm"])
            self.assertEqual(code, 0)
            self.assertIn("DRY RUN", out)
            self.assertIn("WARNING: exe not found", out)
            self.assertIn("WARNING: model not found", out)
            self.assertIn("P128-C520", out)
            self.assertIn("prompt P128-C520 (synthetic): 128 token ids", out)
            self.assertIn("cachewarm", out)
            self.assertEqual(out.count("Total processes"), 1)
            self.assertFalse(out_dir.exists())

    def test_sweep_writes_summary_markdown_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            out_dir = root / "out"
            report = root / "report.md"
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                code, out, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--prompt-lengths", "128,512",
                    "--max-new-tokens", "8", "--repeat", "5",
                    "--source-quant", "q8_0", "--output-dir", str(out_dir),
                    "--tier", "t16", "--device-tag", "RTX_4080_SUPER_16G",
                    "--report-md", str(report)])
            self.assertEqual(code, 0, err)
            summary = json.loads((out_dir / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual([a["arm"] for a in summary["arms"]],
                             ["P128-C520", "P512-C520"])
            arm = summary["arms"][0]
            self.assertEqual(arm["source_quant"], "q8_0")
            self.assertEqual(arm["ttft_kind"], "cold")
            self.assertEqual(arm["repeat"], 5)
            self.assertEqual(arm["stats"]["new"]["n"], 5)
            self.assertEqual(arm["stats"]["new"]["ttft_ms"]["median"], 10.0)
            self.assertEqual(arm["stats"]["new"]["ttft_ms"]["p95"], 10.0)
            self.assertEqual(arm["engine"]["compiled_shaders"], 321)
            attribution = arm["context_attribution"]
            self.assertEqual(attribution["context_bucket_tokens"], 520)
            self.assertEqual(attribution["observed_decode_context_tokens"],
                             128 + 7)
            self.assertEqual(attribution["tpot_ms_per_token"]["median"], 5.0)
            # provenance: GPU + file fingerprints + pre-flight free VRAM
            prov = summary["provenance"]
            self.assertIn("gpu", prov)
            self.assertTrue(prov["exe"]["exists"])
            self.assertEqual(prov["model"]["size_bytes"], 36)
            self.assertIsNotNone(prov["model"]["mtime_utc"])
            self.assertIn("vram_preflight", prov)
            self.assertFalse(summary["context"]["vram_sampling"])
            self.assertIsNone(arm.get("vram"))
            self.assertEqual(summary["context"]["ttft_semantics"],
                             bq.TTFT_SEMANTICS["cold"])
            # summary.md carries the arm + p95, never a single run
            markdown = (out_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("#### P128-C520", markdown)
            self.assertIn("| metric | median | min | p95 | max |", markdown)
            self.assertIn("source_quant: `q8_0`", markdown)
            # the report is generated from the JSON, not retyped
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("| version / arm", report_text)
            self.assertIn("P512-C520", report_text)
            self.assertIn("5 (cold)", report_text)

    def test_vram_preflight_aborts_before_launching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            low = {"gpu_index": 0, "gpu_count": 1, "used_mib": 8000,
                   "free_mib": 64, "total_mib": 8188}
            with mock.patch.object(bq, "query_gpu_memory", return_value=low), \
                    mock.patch.object(bq, "query_gpu_state", return_value=low), \
                    mock.patch.object(bq, "query_gpu_identity",
                                      return_value={"nvidia_smi_available": True,
                                                    "gpu_index": 0,
                                                    "name": "RTX 4060",
                                                    "driver_version": "1",
                                                    "memory_total_mib": 8188,
                                                    "compute_capability": "8.9"}), \
                    mock.patch.object(bq, "run_process") as run:
                code, _, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--repeat", "5",
                    "--min-free-vram-mib", "512",
                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 3)
            self.assertFalse(run.called)
            self.assertIn("ABORTING", err)
            self.assertIn("free VRAM 64 MiB", err)
            self.assertIn("Nothing has been launched", err)

    def test_warm_ttft_runs_a_discarded_process_and_flags_the_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            out_dir = root / "out"
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                code, out, _ = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--repeat", "5",
                    "--ttft-kind", "warm", "--output-dir", str(out_dir)])
            self.assertEqual(code, 0)
            self.assertIn("no in-process prefill repetition",
                          bq.TTFT_SEMANTICS["warm"])
            self.assertIn("[ttft] warm:", out)
            summary = json.loads((out_dir / "summary.json").read_text(
                encoding="utf-8"))
            labels = [c["label"] for c in summary["all_commands"]]
            self.assertTrue(any("cachewarm" in label for label in labels))
            self.assertEqual(len(summary["samples"]), 5)
            self.assertEqual({s["ttft_kind"] for s in summary["samples"]},
                             {"warm"})
            self.assertEqual([s["label"] for s in summary["warmup_samples"]],
                             [label for label in labels if "cachewarm" in label])
            self.assertEqual(summary["context"]["ttft_kind"], "warm")
            self.assertIn("not expressible",
                          summary["context"]["ttft_semantics"])

    def test_max_context_smaller_than_requested_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)

            def truncated(cmd, log_base, *, timeout, label):
                result = fake_run_process(cmd, log_base, timeout=timeout,
                                          label=label)
                result["stdout"] = fake_example_qwen_stdout(
                    cmd, max_context_override="64")
                return result

            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=truncated):
                code, _, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--context-lengths", "2048",
                    "--max-new-tokens", "8", "--repeat", "5",
                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 2)
            self.assertIn("max_context=64", err)
            self.assertIn("smaller than the requested context length 2048", err)

    def test_id_reference_round_trip_through_main(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            reference = root / "ids.json"
            common = ["--exe", str(exe), "--model", str(model),
                      "--prompt-lengths", "128,512",
                      "--context-lengths", "4096", "--max-new-tokens", "8",
                      "--repeat", "5"]
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                code, _, err = quiet_main(
                    common + ["--versions", "dsl-fp8",
                              "--output-dir", str(root / "out1"),
                              "--write-id-reference", str(reference)])
            self.assertEqual(code, 0, err)
            doc = json.loads(reference.read_text(encoding="utf-8"))
            self.assertEqual(sorted(doc["arms"]), ["P128-C4096", "P512-C4096"])

            # a different engine version must still match (that is the point)
            report = root / "report.md"
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                code, _, err = quiet_main(
                    common + ["--versions", "dsl-bf16",
                              "--output-dir", str(root / "out2"),
                              "--require-id-match", str(reference),
                              "--tier", "t16",
                              "--device-tag", "RTX_4080_SUPER_16G",
                              "--report-md", str(report)])
            self.assertEqual(code, 0, err)
            summary = json.loads((root / "out2" / "summary.json").read_text(
                encoding="utf-8"))
            self.assertTrue(all(arm["ids_match"] is True
                                for arm in summary["arms"]))
            self.assertIn("| yes |", report.read_text(encoding="utf-8"))

            # a non-deterministic change is a hard failure naming arm + index
            doc["arms"]["P512-C4096"]["ids"][3] = 424242
            tampered = root / "tampered.json"
            tampered.write_text(json.dumps(doc), encoding="utf-8")
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                code, _, err = quiet_main(
                    common + ["--versions", "dsl-bf16",
                              "--output-dir", str(root / "out3"),
                              "--require-id-match", str(tampered)])
            self.assertEqual(code, 2)
            self.assertIn("P512-C4096", err)
            self.assertIn("index 3", err)

    def test_missing_reference_fails_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            with no_gpu(), mock.patch.object(bq, "run_process") as run:
                code, _, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--repeat", "5",
                    "--require-id-match", str(root / "nope.json"),
                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            self.assertFalse(run.called)
            self.assertIn("cannot read", err)

    def test_report_from_previous_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            out_dir = root / "out"
            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=fake_run_process):
                quiet_main(["--exe", str(exe), "--model", str(model),
                            "--versions", "dsl-fp8", "--repeat", "5",
                            "--tier", "t16",
                            "--device-tag", "RTX_4080_SUPER_16G",
                            "--output-dir", str(out_dir)])
            code, out, err = quiet_main(
                ["--report-from", str(out_dir / "summary.json")])
            self.assertEqual(code, 0, err)
            self.assertIn("| version / arm", out)
            self.assertIn("dsl-fp8 / P1-C64", out)

    def test_report_from_rejects_a_non_benchmark_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "other.json"
            path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
            code, _, err = quiet_main(["--report-from", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("no 'arms'", err)

    def test_bad_arguments_exit_one(self) -> None:
        cases = (
            (["--versions", "nope"], "unknown version"),
            (["--tokens", "a,b"], "comma-separated integer"),
            (["--prompt-lengths", "128", "--repeat", "2"], "for a sweep"),
            (["--profile", "--prompt-lengths", "128", "--repeat", "5"],
             "--profile"),
            (["--prompt-lengths", "512", "--context-lengths", "128"],
             "does not fit"),
            (["--max-argv-chars", "8"], "--max-argv-chars"),
        )
        for argv, message in cases:
            code, _, err = quiet_main(["--dry-run", *argv])
            self.assertEqual(code, 1, f"{argv} should fail")
            self.assertIn(message, err)


# ---------------------------------------------------------------------------
# Feature 7: tiers, tier stamping and the measured-bandwidth roof
# ---------------------------------------------------------------------------

class TierAndRoofTests(unittest.TestCase):
    def args(self, *argv: str):
        return bq.parse_args(["--max-new-tokens", "8", *argv])

    def test_tier_values_are_validated(self) -> None:
        self.assertEqual(bq.resolve_tier("T16"), "t16")
        self.assertEqual(bq.resolve_tier(None), None)
        with self.assertRaisesRegex(bq.BenchmarkError, "--tier must be one of"):
            bq.resolve_tier("t64")

    def test_device_tag_and_bandwidth_are_validated(self) -> None:
        self.assertEqual(bq.resolve_device_tag(" RTX_4080 "), "RTX_4080")
        with self.assertRaisesRegex(bq.BenchmarkError, "must not be empty"):
            bq.resolve_device_tag("   ")
        self.assertEqual(bq.resolve_bandwidth(682.5), 682.5)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(bq.BenchmarkError, "--bandwidth"):
                bq.resolve_bandwidth(bad)

    def test_t16_rejects_bf16_at_128k(self) -> None:
        # 08 section 5: BF16 weights 7.83 GiB + 128 K FP32 KV 8.00 GiB = 16.0 GiB
        # does not fit T16's 15.0 GiB, so BF16 stops at 32 K there ("BF16 at
        # 128 K needs T32"). The old blanket "T16 = 32768" is what this replaces.
        with self.assertRaisesRegex(bq.BenchmarkError, "does not fit tier t16"):
            bq.validate_args(self.args("--tier", "t16", "--versions", "dsl-bf16",
                                       "--prompt-lengths", "131072",
                                       "--max-new-tokens", "32",
                                       "--repeat", "5"))
        # The refusal names the format and its ceiling, not just a number.
        with self.assertRaisesRegex(bq.BenchmarkError,
                                    "bf16 context ceiling"):
            bq.validate_args(self.args("--tier", "t16", "--versions", "dsl-bf16",
                                       "--max-context", "131072",
                                       "--prompt-lengths", "128",
                                       "--max-new-tokens", "32",
                                       "--repeat", "5"))
        # 32 K BF16 on T16 is 10.0 GiB: that is the blessed configuration.
        config = bq.validate_args(self.args("--tier", "t16",
                                            "--versions", "dsl-bf16",
                                            "--prompt-lengths", "32768",
                                            "--max-new-tokens", "32",
                                            "--repeat", "5"))
        self.assertGreaterEqual(config["tier_version_max_context"]["dsl-bf16"],
                                32768 + 32)
        # ... and the same length on T32 stays accepted.
        config = bq.validate_args(self.args("--tier", "t32",
                                            "--versions", "dsl-bf16",
                                            "--prompt-lengths", "32768",
                                            "--max-new-tokens", "32",
                                            "--repeat", "5"))
        self.assertEqual(config["tier"], "t32")
        self.assertEqual(config["tier_max_context"], 262144)

    def test_t16_accepts_the_8_bit_128k_configurations(self) -> None:
        # 08 section 5 + 06 arm A3: T16 reaches 128 K only at 8 bits or less
        # (FP8 12.3 GiB, Q8_0 ~FP8, W4 10.2 GiB). dsl-q8 / dsl-w4 are the seam
        # for 05 Q1/Q2 and must already pass the gate.
        for version in ("dsl-fp8", "dsl-q8", "dsl-w4"):
            config = bq.validate_args(self.args(
                "--tier", "t16", "--versions", version,
                "--prompt-lengths", "131072", "--max-new-tokens", "32",
                "--repeat", "5"))
            self.assertEqual(config["versions"], [version])
            # 128 K of prompt PLUS the decode budget is what the arm allocates,
            # so the ceiling must cover 131072 + 32, not just 131072.
            self.assertGreaterEqual(
                config["tier_version_max_context"][version], 131072 + 32,
                version)

    def test_the_gate_is_per_version_not_per_run(self) -> None:
        # A mixed --versions list must name the version that cannot hold the arm
        # (the default list is dsl-bf16,dsl-fp8).
        with self.assertRaisesRegex(bq.BenchmarkError, "dsl-bf16"):
            bq.validate_args(self.args("--tier", "t16",
                                       "--prompt-lengths", "131072",
                                       "--max-new-tokens", "32",
                                       "--repeat", "5"))

    def test_over_budget_configurations_are_refused_per_format(self) -> None:
        # Every "does not fit" / "no" verdict of 08 section 5, in RATIO terms:
        # the configuration is over budget, not merely above a tier constant.
        for version, capacity in (
                ("dsl-bf16", 131072),   # 7.83 + 8.00 + 0.15 = 15.98 > 15.0
                ("dsl-fp8", 262144),    # 4.17 + 16.00 + 0.15 = 20.32 > 15.0
                ("dsl-q8", 262144),     # ~FP8
                ("dsl-w4", 262144)):    # 2.05 + 16.00 + 0.15 = 18.20 > 15.0
            with self.assertRaisesRegex(bq.BenchmarkError,
                                        "does not fit tier t16"):
                bq.validate_args(self.args(
                    "--tier", "t16", "--versions", version,
                    "--prompt-lengths", "128", "--max-context",
                    str(capacity), "--max-new-tokens", "8", "--repeat", "5"))
        # T32 holds the model's native 262144 in every format.
        for version in ("dsl-bf16", "dsl-fp8", "dsl-q8", "dsl-w4"):
            config = bq.validate_args(self.args(
                "--tier", "t32", "--versions", version, "--prompt-lengths",
                "128", "--max-context", "262144", "--max-new-tokens", "8",
                "--repeat", "5"))
            self.assertEqual(config["tier_max_context"], 262144)

    def test_capacity_matches_the_08_section_5_rows(self) -> None:
        # The budget must reproduce the document's GiB figures, or the gate is
        # arithmetic on invented constants.
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t16", "bf16", 32768)["total_gib"], 10.0,
            delta=0.05)
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t16", "fp8", 131072)["total_gib"], 12.3,
            delta=0.05)
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t16", "w4", 131072)["total_gib"], 14.4,
            delta=0.05)
            # Phase 8b: the SHIPPED W4 policy is plan option (c), W4 decode + a
            # resident FP8 prefill/LM-head copy, so the component table carries BOTH
            # formats (6.24 GiB) instead of the single-copy 2.05 GiB. The single-copy
            # figure (08 section 5's 10.2 GiB) is the explicit --w4-prefill in-kernel
            # arm, kept in W4_IN_KERNEL_PREFILL_WEIGHT_GIB.
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t16", "w4", 262144)["total_gib"], 22.4,
            delta=0.05)
            # Same dual-copy basis as the 128 K row: 22.4 GiB is the option (c)
            # total (single-copy W4 would be 18.2 GiB, 08 section 5).
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t32", "bf16", 262144)["total_gib"], 24.0,
            delta=0.05)
        # BF16 128 K on T16 is over budget by the document's own numbers.
        self.assertGreater(
            bq.tier_capacity_gib("t16", "bf16", 131072)["total_gib"],
            bq.TIER_USABLE_GIB["t16"])

    def test_unknown_weight_format_fails_closed(self) -> None:
        # A version whose format the harness cannot size must not run ungated,
        # and "unknown" must never mean "the tier's maximum".
        self.assertIsNone(bq.weight_format_for_version("mystery"))
        self.assertEqual(bq.tier_format_max_context("t16", None), 32768)
        self.assertEqual(bq.tier_format_max_context("t16", "fp99"), 32768)
        # Fail closed on BOTH tiers: 32768 is the length every KNOWN format
        # holds, while an unknown format could be larger than BF16.
        self.assertEqual(bq.tier_format_max_context("t32", "fp99"), 32768)
        self.assertLess(bq.tier_format_max_context("t32", "fp99"),
                        bq.tier_format_max_context("t32", "bf16"))
        self.assertIsNone(bq.tier_capacity_gib("t16", "fp99", 4096))
        arm = {"key": "P65536-C65536", "prompt_length": 65536,
               "context_length": 65536}
        with self.assertRaisesRegex(bq.BenchmarkError,
                                    "unknown-format context ceiling"):
            bq.check_tier_limits("t16", [arm], max_new_tokens=8,
                                 versions=["mystery"], max_context=65536)
        # Inside the fail-closed ceiling it still runs.
        bq.check_tier_limits("t16", [{"key": "P128-C4096",
                                      "prompt_length": 128,
                                      "context_length": 4096}],
                             max_new_tokens=8, versions=["mystery"],
                             max_context=4096)

    def test_bf16_kv_changes_both_the_budget_and_the_actual_gate(self) -> None:
        self.assertEqual(bq.kv_bytes_per_token("bf16"),
                         bq.kv_bytes_per_token("f32") // 2)
        # fp8 codes are a quarter of FP32 plus 4 B/token of dense block scales.
        self.assertEqual(bq.kv_bytes_per_token("fp8"),
                         bq.kv_bytes_per_token("f32") // 4 + 4)
        self.assertLess(bq.kv_bytes_per_token("fp8"),
                        bq.kv_bytes_per_token("bf16"))
        fp32_cap = bq.tier_format_max_context("t16", "bf16", "f32")
        bf16_cap = bq.tier_format_max_context("t16", "bf16", "bf16")
        self.assertGreater(bf16_cap, fp32_cap)
        self.assertAlmostEqual(
            bq.tier_capacity_gib("t16", "bf16", 131072, "bf16")["kv_gib"],
            4.0, delta=0.01)
        # 128 K BF16-weight + FP32-KV is over T16, but the same weights with
        # BF16 KV fit. The top-level validate_args path must propagate the flag;
        # testing only tier_format_max_context would miss the former omission.
        with self.assertRaisesRegex(bq.BenchmarkError, "does not fit tier t16"):
            bq.validate_args(self.args(
                "--tier", "t16", "--versions", "dsl-bf16", "--kv-dtype", "f32",
                "--prompt-lengths", "131072", "--max-new-tokens", "32",
                "--repeat", "5"))
        config = bq.validate_args(self.args(
            "--tier", "t16", "--versions", "dsl-bf16", "--kv-dtype", "bf16",
            "--prompt-lengths", "131072", "--max-new-tokens", "32",
            "--repeat", "5"))
        self.assertEqual(config["kv_dtype"], "bf16")
        self.assertGreaterEqual(config["tier_version_max_context"]["dsl-bf16"],
                                131072 + 32)

    def test_every_known_version_has_a_capacity_class(self) -> None:
        for version in bq.KNOWN_VERSIONS:
            fmt = bq.weight_format_for_version(version)
            self.assertIsNotNone(fmt, version)
            self.assertIn(fmt, bq.FORMAT_WEIGHT_GIB, version)

    def test_w4_and_q8_versions_are_accepted_before_the_engine_has_them(self):
        # 05 Q1/Q2 seam: the harness must not need a change when the storage
        # appears. An unimplemented --version is the engine's refusal, not ours.
        config = bq.validate_args(self.args("--versions", "dsl-q8,dsl-w4"))
        self.assertEqual(config["versions"], ["dsl-q8", "dsl-w4"])

    def test_model_context_is_gated_without_a_tier(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError,
                                    f"{bq.MODEL_MAX_CONTEXT}-token context"):
            bq.check_tier_limits(None, [{"key": "P262144-C262144",
                                         "prompt_length": 262144,
                                         "context_length": 262144}],
                                 max_new_tokens=8)

    def test_report_md_requires_tier_and_device_tag(self) -> None:
        with self.assertRaisesRegex(bq.BenchmarkError, "--report-md requires"):
            bq.validate_args(self.args("--report-md"))
        with self.assertRaisesRegex(bq.BenchmarkError, "--report-md requires"):
            bq.validate_args(self.args("--tier", "t16", "--report-md"))
        config = bq.validate_args(self.args("--tier", "t16",
                                            "--device-tag", "SKU",
                                            "--report-md"))
        self.assertEqual((config["tier"], config["device_tag"]), ("t16", "SKU"))

    def test_report_refuses_a_summary_without_a_tier_or_device(self) -> None:
        summary = synthetic_summary()
        summary["tier"] = None
        with self.assertRaisesRegex(bq.BenchmarkError, "without a tier"):
            bq.render_report_md(summary)
        summary = synthetic_summary()
        summary["device_tag"] = None
        with self.assertRaisesRegex(bq.BenchmarkError, "without a device tag"):
            bq.render_report_md(summary)

    def test_report_header_carries_tier_and_device(self) -> None:
        text = bq.render_report_md(synthetic_summary())
        self.assertIn("Tier T16 / RTX_4080_SUPER_16G", text)
        row = next(line for line in text.splitlines() if "P128-C4096" in line)
        self.assertIn("T16 / RTX_4080_SUPER_16G", row)

    def test_roof_is_not_available_without_a_measured_bandwidth(self) -> None:
        text = bq.render_report_md(synthetic_summary())
        row = next(line for line in text.splitlines() if "P128-C4096" in line)
        self.assertIn("n/a", row)
        self.assertNotIn("%", row.replace("roof %", ""))

    def test_roof_uses_the_measured_bandwidth(self) -> None:
        summary = synthetic_summary()
        summary["bandwidth_gbps"] = 682.0
        summary["arms"][0]["context_attribution"] = {
            "observed_decode_context_tokens": 135}
        text = bq.render_report_md(summary)
        row = next(line for line in text.splitlines() if "P128-C4096" in line)
        # The fake engine line says storage=fp8 while source_quant is bf16: the
        # byte model must follow the ENGINE storage, because that is what the
        # decode loop streams (a dsl-fp8 arm on a BF16 checkpoint moves FP8).
        # The KV term carries the current 4x GQA read amplification (03 4.1).
        bytes_per_token = bq.roof_bytes_per_token("fp8", 135)
        self.assertAlmostEqual(
            bytes_per_token,
            4.48e9 + 135 * bq.KV_BYTES_PER_TOKEN_FP32
            * bq.DECODE_KV_READ_AMPLIFICATION)
        expected = bq.roof_percent(121.21, bytes_per_token, 682.0)
        self.assertAlmostEqual(expected, 121.21 * bytes_per_token / 682.0e9 * 100)
        self.assertIn(f"{expected:.1f}", row)

    def test_byte_model_counts_the_current_4x_kv_read(self) -> None:
        # 03 section 4.1: the four q_heads of a KV head each re-read the same
        # K/V rows, so a decode step reads 4x the unique KV bytes. Counting it
        # once (the bug this test pins) understates the roof at long context.
        self.assertEqual(bq.DECODE_KV_READ_AMPLIFICATION, 4)
        self.assertEqual(bq.DECODE_KV_READ_AMPLIFICATION_DEDUP, 1)
        unique = 4096 * bq.KV_BYTES_PER_TOKEN_FP32
        self.assertAlmostEqual(bq.kv_bytes_read_per_token(4096),
                               bq.DECODE_KV_READ_AMPLIFICATION * unique)
        self.assertAlmostEqual(bq.kv_bytes_read_per_token(4096, kv_dedup=False),
                               bq.DECODE_KV_READ_AMPLIFICATION * unique)
        # The Phase 5 seam: 1x is what the DEDUPED kernel will read, and it is
        # reachable only by asking for it -- never by default.
        self.assertAlmostEqual(bq.kv_bytes_read_per_token(4096, kv_dedup=True),
                               unique)
        self.assertAlmostEqual(
            bq.roof_bytes_per_token("fp8", 4096, kv_dedup=True),
            4.48e9 + unique)
        self.assertAlmostEqual(
            bq.roof_bytes_per_token("fp8", 4096)
            - bq.roof_bytes_per_token("fp8", 4096, kv_dedup=True),
            (bq.DECODE_KV_READ_AMPLIFICATION - 1) * unique)
        # Weights are still counted once per token.
        self.assertAlmostEqual(bq.roof_bytes_per_token("bf16_packed", 0), 8.41e9)
        # An unknown storage stays n/a in both modes.
        self.assertIsNone(bq.roof_bytes_per_token("mystery", 4096))
        self.assertIsNone(bq.roof_bytes_per_token("mystery", 4096,
                                                 kv_dedup=True))

    def test_roof_model_states_the_assumption_instead_of_claiming_dedup(self):
        note = bq.ROOF_MODEL_NOTE
        self.assertIn(f"{bq.DECODE_KV_READ_AMPLIFICATION}x", note)
        self.assertIn(f"{bq.DECODE_KV_READ_AMPLIFICATION_DEDUP}x", note)
        self.assertIn("03 section 4.1", note)
        self.assertIn("03 section 4.2", note)
        self.assertIn("dedup", note)
        # Both terms are named, and the note says the 1x is conditional on what the
        # engine reported AND on the context that was measured.
        self.assertIn("active=split", note)
        self.assertIn("threshold", note)
        # The rendered per-arm bullet repeats it next to the number it changed.
        line = bq._arm_roof_line({
            "weight_storage": "fp8", "observed_context_tokens": 135,
            "bytes_per_token": bq.roof_bytes_per_token("fp8", 135),
            "kv_read_amplification": bq.DECODE_KV_READ_AMPLIFICATION,
            "kv_dedup_reported_by_engine": None,
            "achieved_gbps": 85.0, "bandwidth_gbps": 682.0, "roof_pct": 12.5})
        self.assertIn("KV read 4x", line)
        self.assertIn("03 section 4.1", line)
        # A Phase-5 arm that DID dedup renders the 1x term and says so.
        deduped = bq._arm_roof_line({
            "weight_storage": "fp8", "observed_context_tokens": 4096,
            "bytes_per_token": bq.roof_bytes_per_token("fp8", 4096,
                                                       kv_dedup=True),
            "kv_read_amplification": bq.DECODE_KV_READ_AMPLIFICATION_DEDUP,
            "kv_dedup": True, "kv_dedup_reported_by_engine": True,
            "kv_split_threshold": 512,
            "achieved_gbps": 85.0, "bandwidth_gbps": 682.0, "roof_pct": 12.5})
        self.assertIn("KV read 1x", deduped)
        self.assertIn("03 section 4.2", deduped)
        # A dedup-capable engine measured below its threshold keeps the 4x term and
        # says WHY (the threshold), never "dedup ran".
        below = bq._arm_roof_line({
            "weight_storage": "fp8", "observed_context_tokens": 135,
            "bytes_per_token": bq.roof_bytes_per_token("fp8", 135),
            "kv_read_amplification": bq.DECODE_KV_READ_AMPLIFICATION,
            "kv_dedup": False, "kv_dedup_reported_by_engine": True,
            "kv_split_threshold": 512,
            "achieved_gbps": 85.0, "bandwidth_gbps": 682.0, "roof_pct": 12.5})
        self.assertIn("KV read 4x", below)
        self.assertIn("below its threshold 512", below)

    def test_decode_attn_dedup_rule_needs_both_the_report_and_the_context(self):
        split = {"decode_attn": {"active": "split", "threshold": 512}}
        # Both conditions are required: an engine report without the context, and a
        # long context without the report, must both stay at the legacy 4x term.
        self.assertTrue(bq.decode_attn_dedup_applies(split, 512))
        self.assertTrue(bq.decode_attn_dedup_applies(split, 4096))
        self.assertFalse(bq.decode_attn_dedup_applies(split, 511))
        self.assertFalse(bq.decode_attn_dedup_applies({"decode_attn": None}, 4096))
        self.assertFalse(bq.decode_attn_dedup_applies(None, 4096))
        self.assertFalse(bq.decode_attn_dedup_applies(
            {"decode_attn": {"active": "legacy", "threshold": 0}}, 4096))
        self.assertFalse(bq.decode_attn_dedup_applies(
            {"decode_attn": {"active": "unavailable", "threshold": 512}}, 4096))
        # A threshold of 0 (no decode-attn line was parsed) can never dedup.
        self.assertFalse(bq.decode_attn_dedup_applies(
            {"decode_attn": {"active": "split", "threshold": 0}}, 4096))
        # The byte model follows the same rule, so the KV term is 1x only above it.
        unique = 4096 * bq.KV_BYTES_PER_TOKEN_FP32
        self.assertAlmostEqual(
            bq.roof_bytes_per_token("fp8", 4096,
                                    kv_dedup=bq.decode_attn_dedup_applies(split, 4096)),
            4.48e9 + unique)
        self.assertAlmostEqual(
            bq.roof_bytes_per_token("fp8", 511,
                                    kv_dedup=bq.decode_attn_dedup_applies(split, 511)),
            4.48e9 + 4 * 511 * bq.KV_BYTES_PER_TOKEN_FP32)

    def test_byte_model_is_documented_and_fails_closed(self) -> None:
        self.assertAlmostEqual(bq.roof_bytes_per_token("bf16_packed", 0), 8.41e9)
        self.assertAlmostEqual(
            bq.roof_bytes_per_token("fp8_packed", 4096),
            4.48e9 + 4096 * bq.KV_BYTES_PER_TOKEN_FP32
            * bq.DECODE_KV_READ_AMPLIFICATION)
        self.assertIsNone(bq.roof_bytes_per_token("f32_dense", 4096))
        self.assertIsNone(bq.roof_bytes_per_token("mystery_quant", 4096))
        self.assertIsNone(bq.roof_percent(100.0, None, 700.0))
        self.assertIsNone(bq.roof_percent(100.0, 8.41e9, None))
        self.assertAlmostEqual(bq.roof_percent(100.0, 1e9, 500.0), 20.0)

    def test_report_from_refuses_a_tier_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "summary.json"
            summary = synthetic_summary()
            path.write_text(json.dumps(summary), encoding="utf-8")
            code, _, err = quiet_main(["--report-from", str(path),
                                       "--report-md",
                                       "--tier", "t32",
                                       "--device-tag", "OTHER"])
            self.assertEqual(code, 1)
            self.assertIn("never mixed", err)
            code, out, err = quiet_main(["--report-from", str(path),
                                         "--report-md"])
            self.assertEqual(code, 0, err)
            self.assertIn("Tier T16", out)


# ---------------------------------------------------------------------------
# Feature 8: long prompts travel as --synthetic-prompt-len / --tokens-file
# ---------------------------------------------------------------------------

class PromptTransportTests(unittest.TestCase):
    def base(self, *extra: str) -> list[str]:
        return ["exe", "--backend", "cuda", "--version", "dsl-fp8",
                "--model", "m.gguf", "--max-new-tokens", "8",
                "--max-context", "8192", "--headless", *extra]

    def test_short_prompt_keeps_the_inline_id_list(self) -> None:
        arm = bq._synthetic_arm(8, 64)
        kind, value = bq.prompt_transport(arm, base_cmd=self.base(),
                                          max_chars=30000)
        self.assertEqual(kind, "tokens")
        self.assertEqual(value, arm["tokens"])

    def test_long_synthetic_prompt_regenerates_in_the_cli(self) -> None:
        arm = bq._synthetic_arm(8192, 8224)
        kind, value = bq.prompt_transport(arm, base_cmd=self.base(),
                                          max_chars=30000)
        self.assertEqual(kind, "synthetic")
        self.assertEqual(value, 8192)

    def test_long_explicit_list_goes_through_a_tokens_file(self) -> None:
        ids = ",".join(str(1000 + i) for i in range(8192))
        arm = {"key": "P8192-C8192", "prompt_length": 8192,
               "context_length": 8192, "tokens": ids, "synthetic": False,
               "origin": "--tokens"}
        kind, _ = bq.prompt_transport(arm, base_cmd=self.base(),
                                      max_chars=30000)
        self.assertEqual(kind, "file")

    def test_explicit_synthetic_ids_still_count_as_synthetic(self) -> None:
        args = bq.parse_args(["--max-new-tokens", "8",
                              "--tokens", bq.ids_to_arg(
                                  bq.synthetic_prompt_ids(4)),
                              "--max-context", "64"])
        arms, _, _ = bq.plan_sweep(args)
        self.assertTrue(arms[0]["synthetic"])

    def test_dry_run_leaves_no_tokens_file_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "never"
            long_ids = ",".join(str(1000 + i) for i in range(8192))
            with no_gpu():
                code, out, _ = quiet_main([
                    "--dry-run", "--exe", str(root / "missing.exe"),
                    "--model", str(root / "missing.gguf"),
                    "--tokens", long_ids, "--max-context", "8224",
                    "--repeat", "2", "--output-dir", str(out_dir)])
            self.assertEqual(code, 0)
            self.assertIn("--tokens-file", out)
            self.assertIn("nothing is written by --dry-run", out)
            self.assertFalse(out_dir.exists())

    def test_the_tokens_file_is_written_before_the_process_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            out_dir = root / "out"
            long_ids = ",".join(str(1000 + i) for i in range(8192))
            seen = {}

            def capture(cmd, log_base, *, timeout, label):
                path = Path(cmd[cmd.index("--tokens-file") + 1])
                seen["exists"] = path.exists()
                seen["ids"] = len(path.read_text(encoding="utf-8").split())
                return fake_run_process(cmd, log_base, timeout=timeout,
                                        label=label)

            with no_gpu(), mock.patch.object(bq, "run_process",
                                             side_effect=capture):
                code, _, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--tokens", long_ids,
                    "--max-context", "8224", "--max-new-tokens", "8",
                    "--repeat", "2", "--output-dir", str(out_dir)])
            self.assertEqual(code, 0, err)
            self.assertTrue(seen["exists"])
            self.assertEqual(seen["ids"], 8192)


# ---------------------------------------------------------------------------
# Feature 9: GPU sampling (utilization/clock/power/temperature) + idle gate
# ---------------------------------------------------------------------------

def gpu_csv(text: str):
    def capture(cmd, *, timeout: float = 20.0) -> str | None:
        return text
    return capture


class GpuSamplingTests(unittest.TestCase):
    def test_full_sample_is_parsed(self) -> None:
        state = bq.query_gpu_state(capture=gpu_csv(
            "0, 2056, 16376, 3, 2745, 42.51, 51\n"))
        self.assertEqual(state["used_mib"], 2056)
        self.assertEqual(state["total_mib"], 16376)
        self.assertEqual(state["utilization_pct"], 3)
        self.assertEqual(state["sm_clock_mhz"], 2745)
        self.assertAlmostEqual(state["power_w"], 42.51)
        self.assertAlmostEqual(state["temperature_c"], 51.0)
        self.assertIn("utilization.gpu", state["fields"])

    def test_unavailable_cells_stay_null(self) -> None:
        state = bq.query_gpu_state(capture=gpu_csv(
            "0, 100, 16376, [N/A], [N/A], [N/A], [N/A]\n"))
        self.assertEqual(state["used_mib"], 100)
        self.assertIsNone(state["utilization_pct"])
        self.assertIsNone(state["sm_clock_mhz"])
        self.assertIsNone(state["power_w"])
        self.assertIsNone(state["temperature_c"])

    def test_no_nvidia_smi_is_none_not_zero(self) -> None:
        self.assertIsNone(bq.query_gpu_state(capture=gpu_csv(None)))

    def test_idle_preflight_statuses(self) -> None:
        idle = bq.query_gpu_state(capture=gpu_csv(
            "0, 2000, 16376, 1, 2745, 40.0, 50\n"))
        ok = bq.preflight_gpu_idle(idle, min_free_mib=512)
        self.assertEqual(ok["status"], "ok")
        busy_state = bq.query_gpu_state(capture=gpu_csv(
            "0, 2000, 16376, 77, 2745, 40.0, 50\n"))
        busy = bq.preflight_gpu_idle(busy_state, min_free_mib=512)
        self.assertEqual(busy["status"], "busy")
        self.assertIn("foreign GPU activity", busy["message"])
        warned = bq.preflight_gpu_idle(busy_state, min_free_mib=512,
                                       enforced=False)
        self.assertEqual(warned["status"], "warning")
        # memory ceiling opt-in
        capped = bq.preflight_gpu_idle(idle, min_free_mib=512,
                                       max_used_mib=1024)
        self.assertEqual(capped["status"], "busy")
        self.assertIn("--gpu-idle-max-used-mib", capped["message"])
        low = bq.preflight_gpu_idle(idle, min_free_mib=15000)
        self.assertEqual(low["status"], "low")
        none = bq.preflight_gpu_idle(None, min_free_mib=512)
        self.assertEqual(none["status"], "unavailable")

    def test_throttle_summary_reports_the_extremes(self) -> None:
        before = bq.query_gpu_state(capture=gpu_csv(
            "0, 100, 16376, 0, 2800, 30.0, 40\n"))
        after = bq.query_gpu_state(capture=gpu_csv(
            "0, 100, 16376, 0, 2600, 30.0, 40\n"))
        during = [{"used_mib": 100, "sm_clock_mhz": 2550, "power_w": 320.5,
                   "temperature_c": 71, "utilization_pct": 98},
                  {"used_mib": 100, "sm_clock_mhz": 2700, "power_w": 280.0,
                   "temperature_c": 65, "utilization_pct": 40}]
        throttle = bq.summarize_throttle(before_state=before,
                                         after_state=after,
                                         during_samples=during)
        self.assertEqual(throttle["min_sm_clock_mhz"], 2550)
        self.assertEqual(throttle["median_sm_clock_mhz"], 2625)
        self.assertEqual(throttle["max_power_w"], 320.5)
        self.assertEqual(throttle["max_temperature_c"], 71)
        self.assertEqual(throttle["sm_clock_change_mhz"], 200)
        self.assertTrue(throttle["before"]["available"])
        # through the VRAM summary the same numbers survive
        summary = bq.summarize_vram_samples(
            before_start_mib=100, during_samples=during, after_exit_mib=50,
            interval_s=0.2, before_state=before, after_state=after)
        self.assertEqual(summary["peak_utilization_pct"], 98)
        self.assertEqual(summary["throttle"]["max_temperature_c"], 71)

    def test_poller_keeps_the_throttle_cells(self) -> None:
        poller = bq.VramPoller(0.02, capture=gpu_csv(
            "0, 4242, 16376, 55, 2600, 300.5, 70\n"))
        poller.start()
        samples = poller.stop()
        self.assertTrue(samples)
        self.assertEqual(samples[0]["used_mib"], 4242)
        self.assertEqual(samples[0]["utilization_pct"], 55)
        self.assertEqual(samples[0]["sm_clock_mhz"], 2600)
        self.assertAlmostEqual(samples[0]["power_w"], 300.5)
        self.assertAlmostEqual(samples[0]["temperature_c"], 70.0)

    def test_busy_gpu_aborts_before_launching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exe, model = make_inputs(root)
            state = {"gpu_index": 0, "used_mib": 1000, "total_mib": 16376,
                     "utilization_pct": 91, "sm_clock_mhz": 2745,
                     "power_w": 300.0, "temperature_c": 60}
            memory = {"gpu_index": 0, "gpu_count": 1, "used_mib": 1000,
                      "free_mib": 15376, "total_mib": 16376}
            with mock.patch.object(bq, "query_gpu_state", return_value=state), \
                    mock.patch.object(bq, "query_gpu_memory", return_value=memory), \
                    mock.patch.object(bq, "query_gpu_identity",
                                      return_value={"nvidia_smi_available": True,
                                                    "gpu_index": 0,
                                                    "name": "RTX 4080 SUPER",
                                                    "driver_version": "1",
                                                    "memory_total_mib": 16376,
                                                    "compute_capability": "8.9"}), \
                    mock.patch.object(bq, "run_process") as run:
                code, _, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--repeat", "5",
                    "--output-dir", str(root / "out")])
            self.assertEqual(code, 3)
            self.assertFalse(run.called)
            self.assertIn("foreign GPU activity", err)
            # the documented opt-out turns the refusal into a warning
            with mock.patch.object(bq, "query_gpu_state", return_value=state), \
                    mock.patch.object(bq, "query_gpu_memory", return_value=memory), \
                    mock.patch.object(bq, "query_gpu_identity",
                                      return_value={"nvidia_smi_available": True,
                                                    "gpu_index": 0,
                                                    "name": "RTX 4080 SUPER",
                                                    "driver_version": "1",
                                                    "memory_total_mib": 16376,
                                                    "compute_capability": "8.9"}), \
                    mock.patch.object(bq, "run_process",
                                      side_effect=fake_run_process):
                code, out, err = quiet_main([
                    "--exe", str(exe), "--model", str(model),
                    "--versions", "dsl-fp8", "--repeat", "5",
                    "--no-idle-check", "--output-dir", str(root / "out2")])
            self.assertEqual(code, 0, err)
            self.assertIn("recorded as a WARNING", out)


# ---------------------------------------------------------------------------
# Feature 10: the C++ generator must match this module's rule exactly
# ---------------------------------------------------------------------------

class GeneratorParityTests(unittest.TestCase):
    """src/qwen/qwen/prompt_ids.h is the CLI's half of --synthetic-prompt-len.

    The constants are parsed from the header and the documented rule is applied
    in Python: if either half changes without the other, ids silently diverge
    and every stored synthetic arm becomes a different prompt. This is the
    cheapest place to catch that (no compiler, no device).
    """

    HEADER = (Path(__file__).resolve().parent.parent
              / "src" / "qwen" / "qwen" / "prompt_ids.h")

    def header_constants(self) -> dict:
        text = self.HEADER.read_text(encoding="utf-8")
        found = {}
        for name in ("kSyntheticPromptSeed", "kSyntheticPromptLengthMix",
                     "kSyntheticPromptLcgMul", "kSyntheticPromptLcgInc",
                     "kSyntheticPromptIdMin", "kSyntheticPromptIdMax"):
            m = re.search(rf"{name}\s*=\s*([0-9A-Fa-fxX]+)[uUlL]*\s*;", text)
            self.assertIsNotNone(m, f"{name} not found in {self.HEADER}")
            found[name] = int(m.group(1), 0)
        return found

    def test_constants_match_the_python_generator(self) -> None:
        c = self.header_constants()
        self.assertEqual(c["kSyntheticPromptSeed"], bq._PROMPT_SEED)
        self.assertEqual(c["kSyntheticPromptLengthMix"],
                         bq._PROMPT_LENGTH_MIX)
        self.assertEqual(c["kSyntheticPromptLcgMul"], bq._PROMPT_LCG_MUL)
        self.assertEqual(c["kSyntheticPromptLcgInc"], bq._PROMPT_LCG_INC)
        self.assertEqual(c["kSyntheticPromptIdMin"], bq.PROMPT_ID_MIN)
        self.assertEqual(c["kSyntheticPromptIdMax"], bq.PROMPT_ID_MAX)
        self.assertLess(bq.PROMPT_ID_MAX, bq.QWEN_VOCAB_SIZE)

    def test_the_documented_rule_reproduces_the_generator(self) -> None:
        c = self.header_constants()
        span = c["kSyntheticPromptIdMax"] - c["kSyntheticPromptIdMin"]
        mask = (1 << 64) - 1
        for length in (1, 8, 128, 512):
            state = (c["kSyntheticPromptSeed"] ^
                     ((length * c["kSyntheticPromptLengthMix"]) & mask)) & mask
            ids = []
            for _ in range(length):
                state = (state * c["kSyntheticPromptLcgMul"] +
                         c["kSyntheticPromptLcgInc"]) & mask
                ids.append(c["kSyntheticPromptIdMin"] + (state >> 32) % span)
            self.assertEqual(ids, bq.synthetic_prompt_ids(length),
                             f"rule diverged for length {length}")


if __name__ == "__main__":
    unittest.main()
