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

def fake_example_qwen_stdout(cmd: list[str], *,
                             max_context_override: str | None = None) -> str:
    """One example_qwen log matching every line benchmark_qwen parses.

    Ids depend only on the prompt length and the decode budget, never on the
    engine version -- that is exactly the property --require-id-match exists to
    prove across versions.
    """
    tokens = cmd[cmd.index("--tokens") + 1]
    prompt_len = len(tokens.split(","))
    max_new = int(cmd[cmd.index("--max-new-tokens") + 1])
    max_context = (max_context_override
                   or cmd[cmd.index("--max-context") + 1])
    ids = [1000 + i * 7 + prompt_len for i in range(max_new)]
    steps = max_new - 1
    return "\n".join([
        f"prompt: {prompt_len} tokens",
        f"prompt ids:{''.join(' ' + t for t in tokens.split(','))}",
        "engine: kernel=dsl storage=fp8 max_context=%s prefill_chunk=64 "
        "gdn=wy-ut-64 cache=12.50 MiB workspace=3.25 MiB shaders=321 "
        "(compile 4.5 ms)" % max_context,
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
            mock.patch.object(bq, "query_gpu_used_mib", return_value=None):
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


if __name__ == "__main__":
    unittest.main()
