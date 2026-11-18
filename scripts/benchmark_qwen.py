#!/usr/bin/env python3
"""Reproducible stdlib-only benchmark driver for example_qwen (real model).

Runs the real Qwen3.5-4B GGUF model through example_qwen SEQUENTIALLY (never
concurrent), preserves every raw stdout/stderr log and benchmark CSV, parses
the CLI's reported metrics, validates determinism (identical generated ids
across repeats), optionally interleaves a baseline executable per repetition
(baseline,new then new,baseline alternating) and computes paired deltas, and
optionally runs one --benchmark-csv profile pass per version.

Normal-mode timing is the CLI's request WALL time, not a pure GPU timer:
  * DSL engines (dsl-bf16 / dsl-fp8): --decode-flight 0 records one command
    list per step and synchronizes its logits download. Default greedy CUDA
    records up to four steps per bounded chunk and synchronizes only the small
    token-history download. Host recording and earlier device work can overlap.
  * Tile DX (tile-* on dx): bounded commit-ONLY command-list flushes every
    sync_every_layers layers (no host wait); one final synchronize at the
    logits download.
  * Tile VK and other stage-sync backends (tile-* on vk, ...): conservative
    stage-boundary commit+SYNCHRONIZE flushes inside blocks and every
    sync_every_layers layers, plus the final download synchronize. The
    normal-mode wall therefore includes backend-dependent batching and
    synchronization; never describe it as uniformly "asynchronous decode".

Profile CSV semantics (from src/qwen/qwen/benchmark.{h,cpp}): every sampled
command is committed and SYNCHRONIZED individually; submit_sync_ms is a
synchronized HOST-WALL interval (submit + full device sync + host overhead),
NOT pure GPU time. Its sum (and every per-category share derived from it)
must never be compared against the normal-mode request wall as if it were
GPU time.

Nothing is fabricated: any missing/malformed CLI output, non-zero exit,
non-deterministic ids, or malformed CSV aborts the whole run with a loud
error. Only --help and --dry-run are safe to exercise without a free GPU.

Usage:
  python scripts/benchmark_qwen.py --help
  python scripts/benchmark_qwen.py --dry-run
  python scripts/benchmark_qwen.py --backend cuda --versions dsl-bf16,dsl-fp8 \
      --baseline-exe path/to/baseline/example_qwen.exe --profile
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KNOWN_VERSIONS = ("dsl-bf16", "dsl-fp8", "tile-bf16", "tile-fp8")

CSV_HEADER = [
    "variant", "backend", "iteration", "ordinal", "phase", "position",
    "rows", "layer", "name", "record_ms", "submit_sync_ms", "total_ms",
]
KNOWN_PHASES = {"reset", "prefill", "decode", "mtp"}

# Broad decode-operator categories. Rules are exact and documented in the
# summary; anything not matched lands in "other" and is listed by name, so
# nothing is silently hidden. The variant (CSV column / --versions entry)
# selects the name grammar; the backend never changes command names (Tile
# DX vs VK differ only in flush/sync policy), so variant alone suffices to
# disambiguate the DSL and Tile naming schemes.
#
# DSL names (dsl-bf16 / dsl-fp8, from dsl_engine.cpp):
# proj.<gguf weight> / proj_res.<gguf weight> (GEMV/GEMM projections)
# lm_head.token_embd.weight (tied LM head)
# input_norm.lN / post_attn_norm.lN / output_norm / gdn_qk_l2norm.lN /
# gdn_gated_norm.lN (norms)
# gdn_recurrent.lN -> recurrence; gdn_conv_silu.lN -> gdn_conv;
# gdn_gates.lN -> gdn_gates; swiglu.lN -> ffn_swiglu
# attn_prepare_rope_kv_append.lN / attn_causal.lN -> attention_core
# ids/positions uploads+copies and logits_download -> transfers
# embedding / reset.zero_* -> other
#
# Tile names (tile-bf16 / tile-fp8, from tile_engine.cpp annotations):
# <proj>[.kseg|.acc|.residual|.store|.store_copy] where <proj> is one of
#   attn.{q,k,v,o}_proj, gdn.{qkv,z,alpha,beta,o}_proj,
#   ffn.{gate,up,down}_proj, head.lm_head   (bare name = single-dispatch
#   GEMM; suffixes = K-split segment/accumulate/epilogue/store-copy kernels)
# attn.prepare_{q,k,v} / attn.rope_{q,k} / attn.kv_append /
#   attn.decode_{dyn,static} / attn.prefill_{dyn,static} -> attention_core
# gdn.conv_silu_step / gdn.conv.gather -> gdn_conv; gdn.gates -> gdn_gates;
#   gdn.recurrent -> recurrence
# gdn.{q,k}_l2norm / gdn.gated_norm / block.input_norm / ffn.post_norm /
#   head.output_norm -> norms; ffn.swiglu -> ffn_swiglu
# upload.{token_ids,rope_positions,attn_len,kv_len} / download.logits /
#   gdn.conv.upload_pos (H2D) and gdn.conv.stage_src (D2D) -> transfers
# embed.gather / block.residual_snapshot / reset.* -> other
CATEGORY_ORDER = [
    "lm_head",
    "ffn_gate/up/down",
    "ffn_swiglu",
    "gdn_projections",
    "attention_projections",
    "attention_core",
    "norms",
    "recurrence",
    "gdn_conv",
    "gdn_gates",
    "transfers",
    "other",
]

_PROJ_RE = re.compile(r"^proj(?:_res)?\.blk\.\d+\.(?P<weight>[a-z_0-9]+)\.weight$")
_LAYER_OP_RE = re.compile(r"^(?P<op>[a-z_0-9]+)\.l\d+$")
_FFN_WEIGHTS = {"ffn_gate", "ffn_up", "ffn_down"}
_GDN_PROJ_WEIGHTS = {"attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"}
_ATTN_PROJ_WEIGHTS = {"attn_q", "attn_k", "attn_v", "attn_output"}
_NORM_OPS = {"input_norm", "post_attn_norm", "gdn_qk_l2norm", "gdn_gated_norm"}
_TRANSFER_NAMES = {
    "ids_upload", "positions_upload", "logits_download",
    "ids_copy", "positions_copy",
}
_KNOWN_OTHER_RE = re.compile(
    r"^(?:embedding|unlabeled|reset\.zero_[a-z_]+\.\d+)$"
)

_TILE_PROJ_RE = re.compile(
    r"^(?P<proj>(?:attn\.(?:q|k|v|o)|gdn\.(?:qkv|z|alpha|beta|o)|"
    r"ffn\.(?:gate|up|down))_proj|head\.lm_head)"
    r"(?P<suffix>\.(?:kseg|acc|residual|store|store_copy))?$"
)
_TILE_ATTN_CORE = {
    "attn.prepare_q", "attn.prepare_k", "attn.prepare_v",
    "attn.rope_q", "attn.rope_k", "attn.kv_append",
    "attn.decode_dyn", "attn.decode_static",
    "attn.prefill_dyn", "attn.prefill_static",
}
_TILE_NORMS = {
    "block.input_norm", "ffn.post_norm", "head.output_norm",
    "gdn.q_l2norm", "gdn.k_l2norm", "gdn.gated_norm",
}
_TILE_TRANSFERS = {
    "upload.token_ids", "upload.rope_positions", "upload.attn_len",
    "upload.kv_len", "download.logits",
    "gdn.conv.upload_pos",  # host->device per-token conv positions
    "gdn.conv.stage_src",   # device->device packed-QKV staging copies
}
_TILE_GDN_CONV = {"gdn.conv_silu_step", "gdn.conv.gather"}
_TILE_KNOWN_OTHER_RE = re.compile(
    r"^(?:embed\.gather|block\.residual_snapshot|unlabeled|"
    r"reset\.(?:k_cache|v_cache|gdn_state|gdn_ring0|gdn_ring1)\.l\d+|"
    r"reset\.(?:zero_operand|conv_src_pad))$"
)


def is_tile_variant(variant: str) -> bool:
    return variant.startswith("tile-")


class BenchmarkError(RuntimeError):
    """Loud, fatal failure: unsupported/malformed output or broken contract."""


def categorize(name: str, variant: str = "") -> str:
    """Map a command name to its broad category. `variant` (dsl-* / tile-*)
    selects the naming scheme; the default keeps the historical DSL rules."""
    if is_tile_variant(variant):
        return _categorize_tile(name)
    if name.startswith("lm_head."):
        return "lm_head"
    m = _PROJ_RE.match(name)
    if m:
        w = m.group("weight")
        if w in _FFN_WEIGHTS:
            return "ffn_gate/up/down"
        if w in _GDN_PROJ_WEIGHTS:
            return "gdn_projections"
        if w in _ATTN_PROJ_WEIGHTS:
            return "attention_projections"
        return "other"
    if name == "output_norm":
        return "norms"
    m = _LAYER_OP_RE.match(name)
    if m:
        op = m.group("op")
        if op in _NORM_OPS:
            return "norms"
        if op == "gdn_recurrent":
            return "recurrence"
        if op == "gdn_conv_silu":
            return "gdn_conv"
        if op == "gdn_gates":
            return "gdn_gates"
        if op == "swiglu":
            return "ffn_swiglu"
        if op in {"attn_prepare_rope_kv_append", "attn_causal"}:
            return "attention_core"
        return "other"
    if name in _TRANSFER_NAMES:
        return "transfers"
    return "other"


def _categorize_tile(name: str) -> str:
    m = _TILE_PROJ_RE.match(name)
    if m:
        proj = m.group("proj")
        if proj == "head.lm_head":
            return "lm_head"
        if proj.startswith("ffn."):
            return "ffn_gate/up/down"
        if proj.startswith("gdn."):
            return "gdn_projections"
        return "attention_projections"
    if name in _TILE_ATTN_CORE:
        return "attention_core"
    if name in _TILE_NORMS:
        return "norms"
    if name == "gdn.recurrent":
        return "recurrence"
    if name in _TILE_GDN_CONV:
        return "gdn_conv"
    if name == "gdn.gates":
        return "gdn_gates"
    if name == "ffn.swiglu":
        return "ffn_swiglu"
    if name in _TILE_TRANSFERS:
        return "transfers"
    return "other"


def _known_other(name: str, variant: str) -> bool:
    if is_tile_variant(variant):
        return bool(_TILE_KNOWN_OTHER_RE.match(name))
    return bool(_KNOWN_OTHER_RE.match(name))


# ---------------------------------------------------------------------------
# CLI output parsing (fail-closed)
# ---------------------------------------------------------------------------

_IDS_RE = re.compile(r"^generated ids \((\d+)\):((?: \d+)*)[ \t]*$", re.MULTILINE)
_TIMING_RE = re.compile(
    r"^timing: host-load (\d+(?:\.\d+)?) ms \| weight-upload (\d+(?:\.\d+)?) ms \| "
    r"engine-compile (\d+(?:\.\d+)?) ms \| TTFT\(prefill (\d+) tokens\) (\d+(?:\.\d+)?) ms"
    r"(?: \| decode (\d+(?:\.\d+)?) ms/token \((\d+) steps\))?[ \t]*$",
    re.MULTILINE,
)
_STOP_RE = re.compile(r"^stop: (.+)$", re.MULTILINE)
_BENCH_RE = re.compile(
    r"^benchmark: commands=(\d+) submits=(\d+) syncs=(\d+) record=(\d+(?:\.\d+)?) ms "
    r"submit\+sync=(\d+(?:\.\d+)?) ms sampled-total=(\d+(?:\.\d+)?) ms "
    r"\(synchronized HOST WALL, not pure GPU time\)[ \t]*$",
    re.MULTILINE,
)
_DECODE_STEP_RE = re.compile(
    r"^decode step=(\d+) input-token=(\d+) wall=(\d+(?:\.\d+)?) ms$", re.MULTILINE
)


def _normalize_newlines(text: str) -> str:
    # Windows text-mode stdout emits CRLF; the raw bytes are preserved on
    # disk, but parsing uses LF so $-anchored patterns behave identically.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_run_output(stdout: str, *, version: str, prompt_len: int,
                     max_new_tokens: int, label: str) -> dict:
    """Parse and validate one example_qwen generation log. Raises on anything
    missing, duplicated, or inconsistent -- never guesses."""
    stdout = _normalize_newlines(stdout)
    ids_matches = _IDS_RE.findall(stdout)
    if len(ids_matches) != 1:
        raise BenchmarkError(
            f"{label}: expected exactly one 'generated ids (N): ...' line, "
            f"found {len(ids_matches)}; unsupported/malformed CLI output")
    declared, ids_str = ids_matches[0]
    ids = [int(x) for x in ids_str.split()]
    if len(ids) != int(declared) or not ids:
        raise BenchmarkError(
            f"{label}: generated-id count mismatch (declared {declared}, "
            f"parsed {len(ids)}) or empty generation")

    timing_matches = _TIMING_RE.findall(stdout)
    if len(timing_matches) != 1:
        raise BenchmarkError(
            f"{label}: expected exactly one 'timing: ...' line, found "
            f"{len(timing_matches)}; unsupported/malformed CLI output")
    (host_load, upload, compile_, prefill_n, ttft,
     decode_ms, decode_steps) = timing_matches[0]
    if int(prefill_n) != prompt_len:
        raise BenchmarkError(
            f"{label}: CLI reports prefill of {prefill_n} tokens but the "
            f"prompt has {prompt_len}; wrong invocation?")

    stop_matches = _STOP_RE.findall(stdout)
    if len(stop_matches) != 1:
        raise BenchmarkError(f"{label}: expected exactly one 'stop:' line")
    stop = stop_matches[0].strip()
    eos = stop.startswith("EOS")
    if not eos and stop != "max-new-tokens budget":
        raise BenchmarkError(f"{label}: unrecognized stop reason {stop!r}")
    if eos:
        if len(ids) > max_new_tokens:
            raise BenchmarkError(f"{label}: EOS run generated more than budget")
    elif len(ids) != max_new_tokens:
        raise BenchmarkError(
            f"{label}: budget stop but generated {len(ids)} != "
            f"max-new-tokens {max_new_tokens}")

    steps = int(decode_steps) if decode_steps else 0
    if decode_ms:
        if steps <= 0:
            raise BenchmarkError(f"{label}: decode ms/token without steps")
        # The CLI decodes once per generated token except the last (and never
        # for the EOS token itself): steps == len(ids) - 1 for a budget stop.
        if not eos and steps != len(ids) - 1:
            raise BenchmarkError(
                f"{label}: decode steps {steps} inconsistent with "
                f"{len(ids)} generated ids (budget stop)")
    elif steps != 0:
        raise BenchmarkError(f"{label}: decode steps without ms/token")

    metrics = {
        "generated_ids": ids,
        "stop": stop,
        "eos": eos,
        "host_load_ms": float(host_load),
        "weight_upload_ms": float(upload),
        "engine_compile_ms": float(compile_),
        "ttft_ms": float(ttft),
        "prefill_tokens": int(prefill_n),
        "decode_ms_per_token": float(decode_ms) if decode_ms else None,
        "decode_steps": steps,
        "decode_tokens_per_s": (1000.0 / float(decode_ms)) if decode_ms else None,
        "version": version,
    }
    return metrics


def parse_benchmark_stdout(stdout: str, label: str) -> dict:
    """Parse the per-iteration 'benchmark:' summary lines of a profile pass."""
    stdout = _normalize_newlines(stdout)
    summaries = []
    for m in _BENCH_RE.finditer(stdout):
        summaries.append({
            "commands": int(m.group(1)),
            "submits": int(m.group(2)),
            "syncs": int(m.group(3)),
            "record_ms": float(m.group(4)),
            "submit_sync_ms": float(m.group(5)),
            "sampled_total_ms": float(m.group(6)),
        })
    steps = [
        {"step": int(m.group(1)), "input_token": int(m.group(2)),
         "wall_ms": float(m.group(3))}
        for m in _DECODE_STEP_RE.finditer(stdout)
    ]
    if not summaries:
        raise BenchmarkError(
            f"{label}: profile pass printed no 'benchmark:' summary lines; "
            f"unsupported/malformed CLI output")
    return {"benchmark_summaries": summaries, "decode_step_walls": steps}


# ---------------------------------------------------------------------------
# Profile CSV parsing / aggregation
# ---------------------------------------------------------------------------

def parse_profile_csv(path: Path, *, version: str, backend: str,
                      label: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise BenchmarkError(f"{label}: profile CSV {path} is empty")
        if header != CSV_HEADER:
            raise BenchmarkError(
                f"{label}: unexpected profile CSV header {header!r}; "
                f"expected {CSV_HEADER!r}")
        rows = []
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(CSV_HEADER):
                raise BenchmarkError(
                    f"{label}: {path}:{lineno}: expected {len(CSV_HEADER)} "
                    f"fields, got {len(row)}")
            rec = dict(zip(CSV_HEADER, row))
            try:
                rec["iteration"] = int(rec["iteration"])
                rec["ordinal"] = int(rec["ordinal"])
                rec["position"] = int(rec["position"])
                rec["rows"] = int(rec["rows"])
                for k in ("record_ms", "submit_sync_ms", "total_ms"):
                    rec[k] = float(rec[k])
            except ValueError as e:
                raise BenchmarkError(
                    f"{label}: {path}:{lineno}: malformed numeric field: {e}")
            if rec["variant"] != version or rec["backend"] != backend:
                raise BenchmarkError(
                    f"{label}: {path}:{lineno}: CSV variant/backend "
                    f"{rec['variant']}/{rec['backend']} != {version}/{backend}")
            if rec["phase"] not in KNOWN_PHASES:
                raise BenchmarkError(
                    f"{label}: {path}:{lineno}: unknown phase {rec['phase']!r}")
            if rec["submit_sync_ms"] < 0 or rec["record_ms"] < 0:
                raise BenchmarkError(f"{label}: {path}:{lineno}: negative timing")
            rows.append(rec)
    if not rows:
        raise BenchmarkError(f"{label}: profile CSV {path} has no samples")
    return rows


def _agg_entry() -> dict:
    return {"count": 0, "submit_sync_ms_sum": 0.0, "record_ms_sum": 0.0,
            "total_ms_sum": 0.0, "submit_sync_ms_min": None,
            "submit_sync_ms_max": None}


def _agg_add(entry: dict, row: dict) -> None:
    v = row["submit_sync_ms"]
    entry["count"] += 1
    entry["submit_sync_ms_sum"] += v
    entry["record_ms_sum"] += row["record_ms"]
    entry["total_ms_sum"] += row["total_ms"]
    entry["submit_sync_ms_min"] = v if entry["submit_sync_ms_min"] is None \
        else min(entry["submit_sync_ms_min"], v)
    entry["submit_sync_ms_max"] = v if entry["submit_sync_ms_max"] is None \
        else max(entry["submit_sync_ms_max"], v)


def aggregate_decode(rows: list[dict], *, expected_iterations: set[int],
                     label: str) -> dict:
    """Group decode-phase raw submit_sync_ms by exact command name and by
    broad category (variant-aware). All values are synchronized HOST-WALL,
    not GPU time."""
    decode = [r for r in rows if r["phase"] == "decode"]
    if not decode:
        raise BenchmarkError(f"{label}: profile CSV contains no decode samples")
    variant = decode[0]["variant"]
    iterations = {r["iteration"] for r in decode}
    if iterations != expected_iterations:
        raise BenchmarkError(
            f"{label}: decode CSV iterations {sorted(iterations)} != expected "
            f"{sorted(expected_iterations)} (one per decode step)")
    by_name: dict[str, dict] = {}
    by_category: dict[str, dict] = {c: _agg_entry() for c in CATEGORY_ORDER}
    per_iteration: dict[int, dict] = {}
    other_names: set[str] = set()
    unrecognized: set[str] = set()
    for r in decode:
        name = r["name"]
        cat = categorize(name, variant)
        by_name.setdefault(name, _agg_entry())
        _agg_add(by_name[name], r)
        _agg_add(by_category[cat], r)
        it = per_iteration.setdefault(
            r["iteration"], {"submit_sync_ms_sum": 0.0, "samples": 0})
        it["submit_sync_ms_sum"] += r["submit_sync_ms"]
        it["samples"] += 1
        if cat == "other":
            other_names.add(name)
            if not _known_other(name, variant):
                unrecognized.add(name)
    for entry in by_name.values():
        entry["submit_sync_ms_mean"] = entry["submit_sync_ms_sum"] / entry["count"]
    total = sum(r["submit_sync_ms"] for r in decode)
    for entry in by_category.values():
        entry["share_of_decode_submit_sync_pct"] = (
            100.0 * entry["submit_sync_ms_sum"] / total if total > 0 else 0.0)
    top = sorted(by_name.items(),
                 key=lambda kv: kv[1]["submit_sync_ms_sum"], reverse=True)
    return {
        "variant": variant,
        "semantics": ("synchronized HOST WALL per command (submit + device "
                      "synchronize + host overhead); NOT pure GPU time. Never "
"compare this sum against the normal-mode request wall "
                 "as if it were GPU time: synchronous DSL decode records one "
                 "command list per step (default greedy CUDA instead syncs "
                 "bounded in-flight chunks); Tile DX uses bounded commit-only "
                      "list flushes plus the final synchronize; Tile VK (and "
                      "other stage-sync backends) additionally synchronizes "
                      "at conservative stage boundaries."),
        "category_share_warning": (
            "Per-category shares are fractions of the SYNCHRONIZED host-wall "
            "sum. Every sample pays a roughly fixed per-command submit+sync "
            "overhead, so categories with many small commands (norms, "
            "transfers, gdn_conv/gdn_gates, attention_core) are inflated "
            "relative to categories with few large ones. These shares are "
            "NOT GPU-time shares and must not be read as such."),
        "decode_sample_count": len(decode),
        "decode_iterations": sorted(iterations),
        "decode_submit_sync_ms_total": total,
        "per_iteration": [
            {"iteration": it, **vals} for it, vals in sorted(per_iteration.items())
        ],
        "by_category": {c: by_category[c] for c in CATEGORY_ORDER},
        "by_name": dict(top),
        "top_operators": [
            {"name": n, **e} for n, e in top[:20]
        ],
        "other_bucket_names": sorted(other_names),
        "unrecognized_names": sorted(unrecognized),
    }


# ---------------------------------------------------------------------------
# Process execution (strictly sequential)
# ---------------------------------------------------------------------------

def build_command(exe: str, version: str, args, csv_path: Path | None) -> list[str]:
    cmd = [
        exe,
        "--backend", args.backend,
        "--version", version,
        "--model", args.model,
        "--tokens", args.tokens,
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-context", str(args.max_context),
        "--headless",
    ]
    if getattr(args, "decode_flight", None) is not None:
        cmd += ["--decode-flight", str(args.decode_flight)]
    if csv_path is not None:
        cmd += ["--benchmark-csv", str(csv_path)]
    return cmd


def run_process(cmd: list[str], log_base: Path, *, timeout: float,
                label: str) -> dict:
    """Run ONE process to completion, preserving raw stdout/stderr bytes."""
    stdout_path = log_base.with_suffix(".stdout.log")
    stderr_path = log_base.with_suffix(".stderr.log")
    print(f"[run] {label}\n      {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            proc = subprocess.run(
                cmd, cwd=str(REPO_ROOT), stdout=out, stderr=err,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            raise BenchmarkError(
                f"{label}: process exceeded {timeout:.0f}s timeout; raw logs: "
                f"{stdout_path} / {stderr_path}")
        except OSError as e:
            raise BenchmarkError(f"{label}: cannot launch {cmd[0]!r}: {e}")
    wall_s = time.monotonic() - t0
    stdout = stdout_path.read_bytes().decode("utf-8", errors="replace")
    stderr = stderr_path.read_bytes().decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((stdout + stderr).splitlines()[-15:])
        raise BenchmarkError(
            f"{label}: exit code {proc.returncode} (expected 0).\n"
            f"raw logs: {stdout_path} / {stderr_path}\n--- output tail ---\n{tail}")
    print(f"[ok ] {label}: exit 0, process wall {wall_s:.1f}s", flush=True)
    return {
        "command": cmd,
        "cwd": str(REPO_ROOT),
        "returncode": proc.returncode,
        "process_wall_s": wall_s,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "ttft_ms", "decode_ms_per_token", "decode_tokens_per_s",
    "host_load_ms", "weight_upload_ms", "engine_compile_ms",
)
_TIME_METRICS = {"ttft_ms", "decode_ms_per_token", "host_load_ms",
                 "weight_upload_ms", "engine_compile_ms"}


def stat_triple(values: list[float]) -> dict:
    return {"median": statistics.median(values),
            "min": min(values), "max": max(values)}


def summarize_samples(samples: list[dict]) -> dict:
    out: dict = {"n": len(samples)}
    for key in _METRIC_KEYS:
        vals = [s["metrics"][key] for s in samples]
        if any(v is None for v in vals):
            if any(v is not None for v in vals):
                raise BenchmarkError(f"inconsistent presence of metric {key}")
            out[key] = None
        else:
            out[key] = stat_triple(vals)
    return out


def paired_deltas(new_samples: list[dict], base_samples: list[dict],
                  version: str) -> dict:
    if len(new_samples) != len(base_samples):
        raise BenchmarkError(f"{version}: unpaired sample counts")
    pairs = []
    for new, base in zip(new_samples, base_samples):
        if new["repetition"] != base["repetition"]:
            raise BenchmarkError(f"{version}: repetition misalignment in pairing")
        if new["metrics"]["generated_ids"] != base["metrics"]["generated_ids"]:
            raise BenchmarkError(
                f"{version} rep {new['repetition']}: baseline/new generated ids "
                f"differ; paired comparison invalid (baseline "
                f"{base['metrics']['generated_ids']} vs new "
                f"{new['metrics']['generated_ids']})")
        deltas = {}
        for key in _METRIC_KEYS:
            nv, bv = new["metrics"][key], base["metrics"][key]
            if nv is None or bv is None:
                deltas[key] = None
                continue
            d = nv - bv
            entry = {"new": nv, "baseline": bv, "delta": d,
                     "pct_change": (100.0 * d / bv) if bv != 0 else None}
            if key in _TIME_METRICS:
                entry["direction"] = ("improvement" if d < 0 else
                                      "regression" if d > 0 else "unchanged")
            deltas[key] = entry
        pairs.append({"repetition": new["repetition"], "deltas": deltas})
    summary = {}
    for key in _METRIC_KEYS:
        vals = [p["deltas"][key] for p in pairs]
        if any(v is None for v in vals):
            summary[key] = None
            continue
        summary[key] = {
            "delta_ms_or_tps": stat_triple([v["delta"] for v in vals]),
            "pct_change": stat_triple([v["pct_change"] for v in vals
                                       if v["pct_change"] is not None])
            if any(v["pct_change"] is not None for v in vals) else None,
        }
    return {"pairs": pairs, "summary": summary}


def check_determinism(samples: list[dict], label: str) -> list[int]:
    ref = samples[0]["metrics"]["generated_ids"]
    for s in samples[1:]:
        if s["metrics"]["generated_ids"] != ref:
            raise BenchmarkError(
                f"{label}: non-deterministic generation across repeats: "
                f"rep {samples[0]['repetition']} ids {ref} vs rep "
                f"{s['repetition']} ids {s['metrics']['generated_ids']}")
    return ref


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_markdown(path: Path, summary: dict) -> None:
    ctx = summary["context"]
    lines = []
    a = lines.append
    a("# example_qwen benchmark summary")
    a("")
    a(f"- Run id: `{summary['run_id']}` (UTC {ctx['started_utc']})")
    a(f"- Working directory: `{ctx['cwd']}`")
    a(f"- Executable: `{ctx['exe']}`")
    if ctx["baseline_exe"]:
        a(f"- Baseline executable: `{ctx['baseline_exe']}`")
    a(f"- Backend: `{ctx['backend']}` | model: `{ctx['model']}`")
    a(f"- Prompt tokens: `{ctx['tokens']}` | max-new-tokens: "
      f"{ctx['max_new_tokens']} | max-context: {ctx['max_context']} | "
      f"repeat: {ctx['repeat']} | warmup: {ctx.get('warmup', 0)} "
      f"(excluded from statistics, correctness-checked)")
    a(f"- Host: {ctx['platform']} | Python {ctx['python']}")
    a("- All processes ran strictly sequentially; raw stdout/stderr logs and "
      "CSVs are preserved under the run directory.")
    a("")
    a("## Complete commands")
    a("")
    for rec in summary["all_commands"]:
        a(f"- `{rec['label']}`: `{' '.join(rec['command'])}`")
    a("")
    a("## Normal-mode results (CLI request-wall timing)")
    a("")
    a("TTFT is the CLI's prefill wall time; decode ms/token is the CLI's mean "
      "decode-step REQUEST WALL time. Batching/synchronization inside that "
"wall is backend-dependent: flight-0 DSL records one command list per "
                 "step and synchronizes its logits download (default greedy CUDA "
                 "synchronizes bounded in-flight chunks); Tile DX flushes bounded "
                 "COMMIT-ONLY command "
      "lists every sync_every_layers layers and synchronizes once at the "
      "download; Tile VK (and other stage-sync backends) performs "
      "conservative commit+SYNCHRONIZE flushes at block-internal stage "
      "boundaries and every sync_every_layers layers, plus the final "
      "download synchronize. This is NOT uniformly 'asynchronous decode'.")
    a("")
    a("tokens/s is the RECIPROCAL of the mean decode ms/token "
      "(1000 / decode_ms_per_token of that run) -- it is NOT the arithmetic "
      "mean of per-sample throughputs, and the median/min/max tokens/s below "
      "are order statistics of those per-run reciprocals.")
    a("")
    for version, vdata in summary["versions"].items():
        a(f"### {version}")
        a("")
        ids = vdata["generated_ids"]
        a(f"Generated ids (identical across all repeats): `{ids}` "
          f"(stop: {vdata['stop']})")
        a("")
        for which, stats in vdata["stats"].items():
            a(f"**{which}** (n={stats['n']})")
            a("")
            a("| metric | median | min | max |")
            a("|---|---|---|---|")
            for key in _METRIC_KEYS:
                s = stats[key]
                if s is None:
                    a(f"| {key} | n/a | n/a | n/a |")
                else:
                    a(f"| {key} | {_fmt(s['median'])} | {_fmt(s['min'])} | "
                      f"{_fmt(s['max'])} |")
            a("")
        if "baseline_comparison" in vdata:
            a("**Paired baseline comparison** (new vs baseline, same "
              "repetition; interleaved baseline,new / new,baseline). "
              "Percent change = (new - baseline) / baseline * 100; for time "
              "metrics a NEGATIVE change is an improvement.")
            a("")
            a("| metric | median delta | median % change | min % | max % |")
            a("|---|---|---|---|")
            for key in _METRIC_KEYS:
                s = vdata["baseline_comparison"]["summary"][key]
                if s is None:
                    a(f"| {key} | n/a | n/a | n/a | n/a |")
                    continue
                d = s["delta_ms_or_tps"]
                p = s["pct_change"]
                a(f"| {key} | {_fmt(d['median'])} | "
                  f"{_fmt(p['median']) if p else 'n/a'} | "
                  f"{_fmt(p['min']) if p else 'n/a'} | "
                  f"{_fmt(p['max']) if p else 'n/a'} |")
            a("")
    if summary["profiles"]:
        a("## Profile passes (--benchmark-csv, one per version)")
        a("")
        a("**WARNING: every profile sample is a synchronized HOST-WALL "
          "interval** (each command is committed and the stream fully "
          "synchronized before the next). `submit_sync_ms` includes submit, "
          "device execution, synchronization and host overhead. The totals "
          "below are NOT pure GPU time and must NEVER be compared against "
          "the normal-mode request wall (see its backend-dependent "
          "batching/sync description above) as if they were.")
        a("")
        a("**WARNING: the per-category 'share %' values are shares of the "
          "SYNCHRONIZED host-wall sum, NOT GPU-time shares.** Each command "
          "pays a roughly fixed submit+synchronize overhead, so categories "
          "made of many small commands (norms, transfers, gdn_conv, "
          "gdn_gates, attention_core) are INFLATED relative to categories "
          "with few large kernels (projections, lm_head). Do not read these "
          "shares as where GPU time goes.")
        a("")
        for version, prof in summary["profiles"].items():
            agg = prof["decode_aggregates"]
            a(f"### {version} decode phase")
            a("")
            a(f"- CSV: `{prof['csv']}`")
            if is_tile_variant(version):
                a("- Tile command labels: bare `<proj>` = single-dispatch "
                  "GEMM; `.kseg`/`.acc`/`.residual`/`.store`/`.store_copy` "
                  "are the K-split segment/accumulate/epilogue/store-copy "
                  "kernels of the SAME projection (grouped with it). "
                  "`gdn.conv.upload_pos` is an H2D copy and "
                  "`gdn.conv.stage_src` a D2D copy (both in transfers).")
                a("")
            a(f"- Decode samples: {agg['decode_sample_count']} across "
              f"{len(agg['decode_iterations'])} decode iterations")
            a(f"- Total decode submit_sync_ms (synchronized host-wall): "
              f"{_fmt(agg['decode_submit_sync_ms_total'])} ms")
            a("")
            a("| category | count | submit_sync sum ms | share % |")
            a("|---|---|---|---|")
            for cat in CATEGORY_ORDER:
                e = agg["by_category"][cat]
                a(f"| {cat} | {e['count']} | {_fmt(e['submit_sync_ms_sum'])} | "
                  f"{_fmt(e['share_of_decode_submit_sync_pct'], 2)} |")
            a("")
            a("Top operators by exact command name (synchronized host-wall):")
            a("")
            a("| name | count | sum ms | mean ms | max ms |")
            a("|---|---|---|---|---|")
            for op in agg["top_operators"]:
                a(f"| `{op['name']}` | {op['count']} | "
                  f"{_fmt(op['submit_sync_ms_sum'])} | "
                  f"{_fmt(op['submit_sync_ms_mean'], 4)} | "
                  f"{_fmt(op['submit_sync_ms_max'], 4)} |")
            a("")
            if agg["other_bucket_names"]:
                a("'other' bucket contains: "
                  + ", ".join(f"`{n}`" for n in agg["other_bucket_names"]))
                a("")
            if agg["unrecognized_names"]:
                a("NOTE: unrecognized command names (kept in 'other', not "
                  "silently dropped): "
                  + ", ".join(f"`{n}`" for n in agg["unrecognized_names"]))
                a("")
    a("## Raw per-process samples")
    a("")
    a("See `summary.json` for every sample's full metrics, exact command, "
      "process wall time and raw log paths. All logs/CSVs are preserved in "
      f"`{summary['run_dir']}`.")
    a("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reproducible sequential benchmark driver for the real "
                    "example_qwen model (stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exe", default="build/windows/x64/release/example_qwen.exe",
                   help="example_qwen executable under test")
    p.add_argument("--baseline-exe", default=None,
                   help="optional baseline executable; interleaved per "
                        "repetition (baseline,new then new,baseline)")
    p.add_argument("--backend", default="cuda", help="backend plugin")
    p.add_argument("--versions", default="dsl-bf16,dsl-fp8",
                   help="comma-separated engine versions")
    p.add_argument("--model", default="data/Qwen3.5-4B-BF16.gguf",
                   help="GGUF checkpoint (BF16; Q8_0 is rejected by the CLI)")
    p.add_argument("--tokens", default="198",
                   help="comma-separated prompt token ids")
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="greedy decode budget per run")
    p.add_argument("--max-context", type=int, default=64,
                   help="KV capacity in tokens")
    p.add_argument("--decode-flight", type=int, default=None,
                   help="optional example_qwen --decode-flight passthrough")
    p.add_argument("--repeat", type=int, default=3,
                   help="repetitions per (version, executable)")
    p.add_argument("--warmup", type=int, default=0,
                   help="warmup runs per (version, executable) executed "
                        "BEFORE the measured repetitions; excluded from "
                        "all statistics/pairing but fully parsed and "
                        "correctness-checked (identical generated ids)")
    p.add_argument("--profile", action="store_true",
                   help="additionally run one --benchmark-csv pass per "
                        "version on --exe (synchronized host-wall profiling)")
    p.add_argument("--output-dir", default=".kimix_cache/qwen_benchmark",
                   help="artifact directory (raw logs preserved per run id)")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="per-process timeout in seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full sequential plan and exit; run nothing")
    return p.parse_args(argv)


def validate_args(args) -> list[str]:
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not versions:
        raise BenchmarkError("--versions is empty")
    for v in versions:
        if v not in KNOWN_VERSIONS:
            raise BenchmarkError(
                f"unknown version {v!r}; supported: {', '.join(KNOWN_VERSIONS)}")
    if len(set(versions)) != len(versions):
        raise BenchmarkError("--versions contains duplicates")
    try:
        token_ids = [int(t) for t in args.tokens.split(",")]
    except ValueError:
        raise BenchmarkError("--tokens must be a comma-separated integer list")
    if not token_ids or any(t < 0 for t in token_ids):
        raise BenchmarkError("--tokens must be non-empty non-negative ids")
    if args.repeat < 1:
        raise BenchmarkError("--repeat must be >= 1")
    if args.warmup < 0:
        raise BenchmarkError("--warmup must be >= 0")
    if args.max_new_tokens < 1:
        raise BenchmarkError("--max-new-tokens must be >= 1")
    if args.decode_flight is not None and args.decode_flight < 0:
        raise BenchmarkError("--decode-flight must be >= 0")
    if len(token_ids) + args.max_new_tokens > args.max_context:
        raise BenchmarkError(
            f"prompt ({len(token_ids)}) + max-new-tokens "
            f"({args.max_new_tokens}) exceeds max-context ({args.max_context})")
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    return versions


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        versions = validate_args(args)
    except BenchmarkError as e:
        print(f"benchmark_qwen: error: {e}", file=sys.stderr)
        return 1

    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = REPO_ROOT / exe_path
    baseline_path = None
    if args.baseline_exe:
        baseline_path = Path(args.baseline_exe)
        if not baseline_path.is_absolute():
            baseline_path = REPO_ROOT / baseline_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    run_dir = out_dir / run_id

    # ---- plan -------------------------------------------------------------
    plan = []  # (label, exe, version, kind, repetition, csv_path|None)
    for w in range(args.warmup):
        order = (["baseline", "new"] if baseline_path else ["new"])
        for version in versions:
            for which in order:
                exe = str(baseline_path if which == "baseline" else exe_path)
                label = f"{version}/{which}/warmup{w}"
                plan.append((label, exe, version, "warmup", w, None))
    for rep in range(args.repeat):
        order = (["baseline", "new"] if baseline_path else ["new"])
        if baseline_path and rep % 2 == 1:
            order = ["new", "baseline"]
        for version in versions:
            for which in order:
                exe = str(baseline_path if which == "baseline" else exe_path)
                label = f"{version}/{which}/rep{rep}"
                plan.append((label, exe, version, "normal", rep, None))
    if args.profile:
        for version in versions:
            csv_path = run_dir / f"profile_{version}.csv"
            plan.append((f"{version}/profile", str(exe_path), version,
                         "profile", None, csv_path))

    if args.dry_run:
        print("DRY RUN -- nothing will be executed. Sequential plan:")
        print(f"  cwd: {REPO_ROOT}")
        print(f"  output dir: {out_dir} (run dir {run_dir})")
        for warning, path in (("exe", exe_path), ("model", model_path)):
            if not path.exists():
                print(f"  WARNING: {warning} not found: {path}")
        if baseline_path and not baseline_path.exists():
            print(f"  WARNING: baseline exe not found: {baseline_path}")
        for i, (label, exe, version, kind, rep, csv_path) in enumerate(plan, 1):
            cmd = build_command(exe, version, args, csv_path)
            print(f"  {i:2d}. [{label}] {' '.join(cmd)}")
        print(f"Total processes: {len(plan)} (strictly sequential)")
        return 0

    for what, path in (("executable", exe_path), ("model", model_path)):
        if not path.exists():
            print(f"benchmark_qwen: error: {what} not found: {path}",
                  file=sys.stderr)
            return 1
    if baseline_path and not baseline_path.exists():
        print(f"benchmark_qwen: error: baseline executable not found: "
              f"{baseline_path}", file=sys.stderr)
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_len = len(args.tokens.split(","))
    started_utc = datetime.now(timezone.utc).isoformat()

    samples: dict = {v: {"new": [], "baseline": []} for v in versions}
    warmups: dict = {v: {"new": [], "baseline": []} for v in versions}
    profiles: dict = {}
    all_commands = []

    try:
        for label, exe, version, kind, rep, csv_path in plan:
            cmd = build_command(exe, version, args, csv_path)
            log_base = run_dir / label.replace("/", "_")
            result = run_process(cmd, log_base, timeout=args.timeout,
                                 label=label)
            all_commands.append({"label": label, "command": cmd})
            metrics = parse_run_output(
                result["stdout"], version=version, prompt_len=prompt_len,
                max_new_tokens=args.max_new_tokens, label=label)
            sample = {
                "label": label, "repetition": rep, "kind": kind,
                "command": cmd, "cwd": result["cwd"],
                "returncode": result["returncode"],
                "process_wall_s": result["process_wall_s"],
                "stdout_log": result["stdout_log"],
                "stderr_log": result["stderr_log"],
                "metrics": metrics,
            }
            if kind in ("normal", "warmup"):
                bucket = samples if kind == "normal" else warmups
                bucket[version]["baseline" if "baseline" in label
                                else "new"].append(sample)
            else:
                bench = parse_benchmark_stdout(result["stdout"], label)
                rows = parse_profile_csv(csv_path, version=version,
                                         backend=args.backend, label=label)
                steps = metrics["decode_steps"]
                expected = set(range(1, steps + 1))
                agg = aggregate_decode(rows, expected_iterations=expected,
                                       label=label)
                if len(bench["benchmark_summaries"]) != steps + 1:
                    raise BenchmarkError(
                        f"{label}: expected {steps + 1} 'benchmark:' summaries "
                        f"(prefill + {steps} decode), got "
                        f"{len(bench['benchmark_summaries'])}")
                profiles[version] = {
                    "sample": sample,
                    "csv": str(csv_path),
                    "stdout_benchmark": bench,
                    "decode_aggregates": agg,
                }

        # ---- determinism + pairing ----------------------------------------
        # Warmup runs are correctness-checked here (same parsing already ran
        # above, plus id determinism) but NEVER enter stats or pairing.
        version_reports = {}
        for version in versions:
            vdata: dict = {"stats": {}}
            ref_ids = None
            for which in ("new", "baseline"):
                ws = warmups[version][which]
                if ws:
                    wids = check_determinism(ws, f"{version}/{which}/warmup")
                    if ref_ids is not None and wids != ref_ids:
                        raise BenchmarkError(
                            f"{version}: warmup and measured runs generate "
                            f"different ids ({wids} vs {ref_ids}); "
                            f"non-deterministic generation, run invalid")
                    ref_ids = wids
            for which in ("new", "baseline"):
                ss = samples[version][which]
                if not ss:
                    continue
                ids = check_determinism(ss, f"{version}/{which}")
                if ref_ids is not None and ids != ref_ids:
                    raise BenchmarkError(
                        f"{version}: baseline and new executables generate "
                        f"different ids ({ref_ids} vs {ids}); comparison invalid")
                ref_ids = ids
                vdata["stats"][which] = summarize_samples(ss)
            vdata["generated_ids"] = ref_ids
            vdata["warmup_checked"] = {
                which: len(warmups[version][which])
                for which in ("new", "baseline") if warmups[version][which]
            }
            vdata["stop"] = samples[version]["new"][0]["metrics"]["stop"]
            if baseline_path:
                vdata["baseline_comparison"] = paired_deltas(
                    samples[version]["new"], samples[version]["baseline"],
                    version)
            if version in profiles:
                prof_ids = profiles[version]["sample"]["metrics"]["generated_ids"]
                if prof_ids != ref_ids:
                    raise BenchmarkError(
                        f"{version}: profile pass generated different ids "
                        f"({prof_ids}) than normal runs ({ref_ids}); "
                        f"benchmark mode must not change greedy output")
            version_reports[version] = vdata

        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "context": {
                "started_utc": started_utc,
                "cwd": str(REPO_ROOT),
                "exe": str(exe_path),
                "baseline_exe": str(baseline_path) if baseline_path else None,
                "backend": args.backend,
                "model": str(model_path),
                "model_arg": args.model,
                "tokens": args.tokens,
                "max_new_tokens": args.max_new_tokens,
        "max_context": args.max_context,
        "decode_flight": args.decode_flight,
        "repeat": args.repeat,
                "warmup": args.warmup,
                "profile": args.profile,
                "timeout_s": args.timeout,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "execution": "strictly sequential, one process at a time",
            },
            "all_commands": all_commands,
            "versions": version_reports,
            "samples": samples,
            "warmup_samples": warmups,
            "profiles": profiles,
        }
        summary_json = out_dir / "summary.json"
        summary_md = out_dir / "summary.md"
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(summary_md, summary)
        # Keep a copy inside the run dir so every run is self-contained.
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(run_dir / "summary.md", summary)
        print(f"\nSummary written: {summary_json}")
        print(f"                 {summary_md}")
        print(f"Raw artifacts:   {run_dir}")
        return 0
    except BenchmarkError as e:
        print(f"\nbenchmark_qwen: FAILED: {e}", file=sys.stderr)
        print(f"Raw artifacts preserved in: {run_dir}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
