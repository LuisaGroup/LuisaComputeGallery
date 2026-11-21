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

Sweeps (opt_plan/06-benchmark-plan.md sections 4, 5, 7, 8)
----------------------------------------------------------
--prompt-lengths 128,512,2048 synthesizes token-id lists and drives the
existing --tokens argument. A UTF-8 text prompt is NEVER used: its token count
depends on the tokenizer, so the prefill length would silently change. The
generator is a 64-bit LCG seeded from the requested length (see
synthetic_prompt_ids and the rule in --prompt-lengths' help), so the same
length yields byte-identical ids on every run, every machine and every Python
version; ids are drawn from [PROMPT_ID_MIN, PROMPT_ID_MAX) = [1000, 200000),
a range that avoids the low control ids and the high special-token block of
Qwen3.5-4B (vocab 248320) and is validated by the CLI itself ("token id ...
outside vocab"). Every arm label records its prompt length and context length
("P<prompt>-C<context>"), so a number can always be attributed to an arm.

Long prompts and the Windows command-line cap (--synthetic-prompt-len /
--tokens-file)
-------------------------------------------------------------------------
A Windows command line is capped at 32767 characters and an id list costs ~6
characters per token, so a sweep with P >= 8192 cannot pass --tokens <ids>.
The harness therefore picks the cheapest transport that the CLI supports, in
this order:
  1. --tokens <ids>              when the whole command line fits;
  2. --synthetic-prompt-len <P>  when the arm's ids ARE synthetic_prompt_ids(P)
                                 (every sweep arm): the CLI regenerates the
                                 exact same list with the same LCG constants
                                 (src/qwen/qwen/prompt_ids.h);
  3. --tokens-file <path>        otherwise (an explicit --tokens list that is
                                 too long): the ids are written to
                                 <run dir>/tokens_<arm>.txt immediately before
                                 that process is launched, so --dry-run still
                                 touches nothing.
Every run records which transport it used (command + summary). The id lists are
identical in all three cases; scripts/test_benchmark_qwen.py cross-checks the
C++ generator constants against this module.

Tiers, roof bandwidth and mixing rules (--tier / --device-tag / --bandwidth)
--------------------------------------------------------------------------
08-target-hardware.md section 1: a T16 number and a T32 number are two
experiments and never share a table. --tier t16|t32 and --device-tag <sku> are
stamped into every sample, arm record and summary; --report-md REFUSES to render
a table without both, and --report-from refuses a stored summary that lacks them
(or whose tier differs from the --tier given on the command line). With --tier
the requested prompt/context lengths are gated against 08 section 5, whose
verdict is per (tier, weight FORMAT), not per tier: T16 reaches 128 K only at 8
bits or less (FP8/Q8_0 174981 tokens, W4 209715), because 7.83 GiB of BF16
weights plus 8.00 GiB of 128 K FP32 KV exceed its 15.0 GiB (BF16 stops at
115015), while T32 holds the model's native 262144 in every format. The gate
checks the prompt plus the decode budget AND the --max-context actually passed,
and a version whose weight format it does not know fails closed to the length
every format holds; no run may exceed the model's 262144.

--bandwidth <GB/s> is the MEASURED achievable DRAM bandwidth from the
calibration (tests/benchmark/benchmark_qwen_decode --calibrate, 08 section 7
step 7), in decimal GB/s. It is the only denominator the report will use for a
"roof %" column: without it every roof value is `n/a`, never a vendor number.
The per-token byte model is documented, not guessed (08 section 2.1/2.2 plus
03 section 4.1): weights bf16_packed 8.41e9 B / fp8_packed 4.48e9 B / q8_0
4.47e9 B -- keyed by the ENGINE's weight storage from its `engine:` line,
because a dsl-fp8 arm on a BF16 checkpoint streams FP8 bytes -- plus FP32 KV
65536 B per token per attended position times the CURRENT 4x GQA read
amplification of the per-q_head decode attention (GQA dedup/split-KV is NOT
implemented, so the model never claims the deduped 1x); an unknown storage
yields n/a. DECODE_KV_READ_AMPLIFICATION is the single seam that switches the
KV term to 1x when Phase 5 lands dedup and the engine reports it.

GPU sampling, idle pre-flight and throttle probe (--vram / --idle-check)
------------------------------------------------------------------------
`nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw,
temperature.gpu` is sampled before, during and after a run. The pre-flight
refuses to launch (exit 3, 08 section 9 H3) when another tenant is using the
GPU: utilization.gpu above --gpu-idle-util-pct (default 5) or less free VRAM
than --min-free-vram-mib; a pre-run memory.used ceiling can be enforced with
--gpu-idle-max-used-mib. --no-idle-check downgrades the utilization refusal to a
recorded warning (for machines whose desktop compositor keeps the GPU slightly
busy) and is stated in the summary. Before/after clock, power and temperature
samples are always recorded as the throttle probe of 08 section 7 step 8, and
the in-flight samples keep the minimum SM clock, maximum power draw and maximum
temperature, so a 5-15 % sustained-clock drop is visible instead of implied.
Without nvidia-smi every field stays null and the run continues.

--context-lengths 128,512,2048,8192 attributes TPOT (ms/token) to a context
bound. The CLI prints one ms/token per run, so each value C is measured with a
prefill of C tokens: the decode loop then runs at ctx ~= C (plan section 4:
"repeat with C and the prefill length chosen so the decode happens at
ctx"). --max-context is raised to cover the largest requested value plus the
decode budget and is recorded per arm; if the binary reports a smaller
max_context in its own "engine:" line the run aborts loudly.

TTFT (--ttft-kind cold|warm)
----------------------------
cold = one process per repetition with a fresh engine (weight upload is inside
TTFT, shader compilation happens before it and is reported separately).
example_qwen performs exactly ONE prefill per process and has no in-process
prefill repetition, so a true in-process "warm" prefill is NOT expressible
with the current CLI. --ttft-kind warm is therefore implemented honestly as:
each measured repetition runs after ONE discarded first process of the same
arm and executable. The engine is created again (weight upload and engine
creation stay inside TTFT); only the on-disk shader/pipeline cache, the page
cache and the driver caches are warm. Such samples live in their own
ttft_kind bucket, are never mixed with cold samples, and the log line states
this explicitly. ttft_kind is recorded in the run context, in every arm and in
every sample.

VRAM (--vram)
-------------
Free VRAM is read before launch and the run aborts (exit 3) when it is below
--min-free-vram-mib (default 512) -- nothing has been launched at that point.
`nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw,
temperature.gpu --format=csv,noheader,nounits` is then polled every
--vram-interval seconds while the process runs. The sampling is COARSE and
labelled as such: each sample is a separate nvidia-smi process, so the peak is a
sampled maximum, and nvidia-smi cannot isolate the instant after prefill -- that
checkpoint is reported as a coarse upper bound of the running resident set.
Without nvidia-smi every VRAM field is null and the run continues, so the
harness stays usable on a machine with no GPU.

ID references (--write-id-reference / --require-id-match)
--------------------------------------------------------
A reference stores the greedy ids per arm so an optimization can be proven not
to change the model. Schema (benchmark_qwen.id_reference v1):

  {"schema": "benchmark_qwen.id_reference",
   "version": 1,                     # SCHEMA version (int), not the engine
   "generated_utc": "....+00:00",
   "backend": "cuda",
   "model": "data/Qwen3.5-4B-BF16.gguf",
   "source_quant": "bf16",
   "max_new_tokens": 8,
   "note": "....",
   "arms": {
     "P128-C4096": {"prompt_length": 128, "context_length": 4096,
                    "tokens": 8, "ids": [....], "stop": "....",
                    "engine_version": "dsl-fp8"}}}

Arms are keyed by "P<prompt_length>-C<context_length>", so a reference is
executable-, backend- and quant-independent (that is the point: arm A7 vs arm
A0 must produce the same ids). --require-id-match aborts non-zero naming the
arm and the first differing token index, and also aborts when the run contains
an arm the reference does not cover (fail closed: a missing arm can never pass
silently).

Markdown report (--report-md [PATH] / --report-from SUMMARY.json)
-----------------------------------------------------------------
--report-md renders the table used by docs/examples/example_qwen.md directly
from the collected summary JSON -- numbers are generated, never retyped. No
PATH means stdout, a PATH means that file. --report-from SUMMARY.json rebuilds
the same table offline from an earlier summary.json. Both refuse to render
without a tier and a device tag (table header of 08 section 1), and the roof
column is `n/a` unless the run (or the command line) carried a measured
--bandwidth.

Provenance
----------
Every number in summary.json carries where it came from: GPU name / compute
capability / driver version, free VRAM before start, binary and model file size
and mtime, the per-run engine lines (compiled shader count, cuda-graph launches
and updates, cache/workspace bytes) and a mandatory source_quant ("bf16",
"q8_0", ...) so a double-quantized Q8_0-sourced arm can never be silently mixed
with a BF16-sourced FP8 arm (plan risk B3). --source-quant sets it; without the
flag it is inferred from an explicit "BF16"/"Q8_0" file-name marker and the
inference itself is recorded ("source_quant_origin"). A value this harness does
not know is recorded verbatim with origin "flag-unverified" instead of being
rejected, so a future quantizer needs no harness change.

Statistics
----------
No single run is ever reported: --repeat defaults to 3 and a sweep requires at
least 5 repetitions (SWEEP_REPEAT_DEFAULT), refusing less rather than quietly
reporting one sample. Every metric reports median, min, p95 and max; p95 uses
linear interpolation (percentile()) and is labelled as interpolated, because
with 5 samples it is not an observed repetition.

Exit codes: 0 success, 1 bad arguments / missing inputs, 2 the run failed
(malformed output, non-zero exit, id mismatch, ...), 3 refused by the VRAM
pre-flight before anything was launched.

Usage:
  python scripts/benchmark_qwen.py --help
  python scripts/benchmark_qwen.py --dry-run
  python scripts/benchmark_qwen.py --backend cuda --versions dsl-bf16,dsl-fp8 \
      --baseline-exe path/to/baseline/example_qwen.exe --profile
  python scripts/benchmark_qwen.py --backend cuda --versions dsl-fp8 \
      --prompt-lengths 128,512,2048 --max-new-tokens 32 --repeat 5 --vram \
      --source-quant bf16 --write-id-reference .kimix_cache/qwen_benchmark/ids_bf16.json
  python scripts/benchmark_qwen.py --backend cuda --versions dsl-fp8 \
      --context-lengths 128,512,2048 --max-new-tokens 32 --require-id-match \
      .kimix_cache/qwen_benchmark/ids_bf16.json --report-md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Versions this harness accepts. dsl-q8 (05 Q1) and dsl-w4 (05 Q2) are the plan's
# arms A2/A3/A4: they exist as a SEAM (their capacity class is already in
# VERSION_WEIGHT_FORMAT, so the tier gate reasons about them correctly), and the
# engine rejects a --version it has not implemented, which fails closed instead of
# silently benchmarking another format.
KNOWN_VERSIONS = ("dsl-bf16", "dsl-fp8", "tile-bf16", "tile-fp8",
                  "dsl-q8", "dsl-w4")

# ---------------------------------------------------------------------------
# Contract constants (see the module docstring for the why of each)
# ---------------------------------------------------------------------------

ID_REFERENCE_SCHEMA = "benchmark_qwen.id_reference"
ID_REFERENCE_VERSION = 1

# Never report a single run: a plain run needs >= 2 repetitions, a sweep needs
# >= SWEEP_REPEAT_DEFAULT (plan risk B4: median + p95 on a shared desktop GPU).
MIN_REPEAT = 2
SWEEP_REPEAT_DEFAULT = 5
DEFAULT_REPEAT = 3

DEFAULT_MIN_FREE_VRAM_MIB = 512
DEFAULT_VRAM_INTERVAL_S = 0.2

# Windows CreateProcess caps the whole command line at 32767 characters. The
# harness checks the projected command line BEFORE launching and, when it does
# not fit, moves the ids off the command line (--synthetic-prompt-len for a
# synthetic arm, --tokens-file otherwise: opt_plan/09 N4.1). This limit is only
# the final guard; exceeding it now means even that fallback failed.
DEFAULT_MAX_ARGV_CHARS = 30000

# ---------------------------------------------------------------------------
# Tier / weight-format capacity, roof bandwidth and the decode byte model
# (08-target-hardware.md sections 1, 2, 5, 7; 03-decode-latency.md sections 1, 4)
# ---------------------------------------------------------------------------

TIERS = ("t16", "t32")
# Usable VRAM per tier (08 section 1; 1 GiB is reserved for the desktop/WDDM).
TIER_USABLE_GIB = {"t16": 15.0, "t32": 31.0}
# The model's own limit (example_qwen --max-context <= 262144).
MODEL_MAX_CONTEXT = 262144
GIB = float(1 << 30)

# Capacity is a property of the (tier, weight FORMAT) pair, NOT of the tier.
# 08 section 5's verdict list is per configuration: "BF16, ctx 128 K -> 16.0 GiB,
# does not fit T16", "FP8/Q8_0, ctx 128 K -> 12.3 GiB, fits", "W4, ctx 128 K ->
# 10.2 GiB, fits", "BF16/FP8/W4, ctx 262144 -> fits on T32", "W4, ctx 262144 ->
# 18.2 GiB, no on T16". So T16 reaches 128 K only at 8 bits or less, T32 reaches
# the model's native 262144 in every format, and a single "T16 = 32768" constant
# is wrong for both directions (it refused the 8-bit 128 K configuration arm A3
# is built on, while stating no reason a 32 K BF16 run is the ceiling).
#
# The component rows of that same section are the model below, in GiB:
#   weights  BF16 7.83 | FP8/Q8_0 4.17 | W4 g128 2.05
#   KV (FP32) 0.25 at ctx 4 K | 2.00 at 32 K | 8.00 at 128 K  = 64 KiB/token
#   GDN state + scratch + logits/workspace/context  ~0.15
#   usable  T16 15.0 | T32 31.0
# Solving it for the token count reproduces every row of 08 section 5 (BF16
# 32 K -> 9.98 GiB, FP8 128 K -> 12.32, W4 128 K -> 10.20, W4 262144 -> 18.20,
# BF16 262144 -> 23.98).
FORMAT_WEIGHT_GIB = {
    # 08 section 5 "text-trunk weights"; the byte figures are 08 section 2.1.
    "bf16": 7.83, # 8.41e9 B
    "fp8": 4.17, # 4.48e9 B, FP8 E4M3 + scales
    "q8_0": 4.41, # measured 4514.92 MiB device: codes + FP32-widened scales
    # Phase 8b plan option (c), the SHIPPED W4 policy: W4 decode weights PLUS a
    # resident FP8 prefill/LM-head copy. The device therefore holds both formats:
    # 2.05 GiB of W4 (int4 codes + one FP32 scale per 128 K) + 4.17 GiB of FP8
    # (codes + one FP32 scale per 64 K) = 6.22 GiB, plus the scale arrays counted
    # once each (~0.02 GiB) -> 6.24 GiB. A single-copy W4 session
    # (--w4-prefill in-kernel, the plan option (b) A/B arm) is 2.05 GiB and is
    # NOT what the tier verdict covers: the default is (c).
    "w4": 6.24,
}
# Single-copy W4 (--w4-prefill in-kernel, plan option (b)): reporting only.
W4_IN_KERNEL_PREFILL_WEIGHT_GIB = 2.05
# --w4-pack is required by the dsl-w4 engine variant; the harness never invents
# one (a pack costs minutes of offline clipping search, see
# scripts/qwen_quality_gate.py and example_qwen --export-w4-pack).
FIXED_OVERHEAD_GIB = 0.15   # GDN state + conv history + scratch + workspace
# KV bytes per token of context, keyed by the cache's STORED dtype
# (opt_plan/05-quantization.md Q3). The FP32 entry is the plan's model: 8
# full-attention layers x 2 (K and V) x 4 kv heads x 256 head_dim x 4 B = 64 KiB.
# BF16 stores two RNE codes per 32-bit word, so the same payload is 32 KiB/token --
# the term that decides the T16 capacity line (08 section 5) and the decode roof
# from ctx ~8 K up (03 section 4.3). The blocked 64-position layout adds at most
# 63 * head_dim codes of padding per head to the RESERVATION, never to the traffic.
KV_BYTES_PER_TOKEN_BY_DTYPE = {
    "f32": 65536,
    "bf16": 32768,
    # fp8_e4m3: four codes per 32-bit word in the 64-position blocked layout, so the
    # code payload is a quarter of FP32. The cache ALSO keeps one dense FP32 scale
    # per (kv_head, 64-position block): 8 layers x 2 regions x 4 heads / 64 positions
    # = 1 float per token per... see below -- 4 bytes per token of context, i.e.
    # 1/4096 of the FP32 payload. It is charged here so the capacity model does not
    # hide a real allocation (the allocator reports the same number as
    # QwenEngineInfo::kv_scale_bytes).
    "fp8": 16384 + 4,
}
KV_BYTES_PER_TOKEN_FP32 = KV_BYTES_PER_TOKEN_BY_DTYPE["f32"]


def kv_bytes_per_token(kv_dtype: str | None = "f32") -> int:
    """Stored KV bytes per token of context for a cache dtype.

    Unknown/None keeps the FP32 model (the legacy default and the only dtype the
    capacity table is calibrated on) instead of inventing a number.
    """
    return KV_BYTES_PER_TOKEN_BY_DTYPE.get(kv_dtype or "f32",
                                          KV_BYTES_PER_TOKEN_BY_DTYPE["f32"])
# 08 section 5 has no verdict for a format outside FORMAT_WEIGHT_GIB (unquantized
# F32 weights, a format that does not exist yet), so neither its weight bytes nor
# its ceiling are known. Such a run is allowed only up to the length EVERY known
# format holds on ANY tier -- 32768, "T16 is a 32 K-context machine in every
# format" -- on BOTH tiers, not just on T16. Fail closed: an unknown format never
# means "the tier's maximum". Give it a FORMAT_WEIGHT_GIB row (and a
# VERSION_WEIGHT_FORMAT entry) to unlock longer contexts; a test keeps the two
# tables in sync, so a new version cannot silently land in this bucket.
FORMAT_UNKNOWN_MAX_CONTEXT = {"t16": 32768, "t32": 32768}

# Which weight format each ENGINE version streams on the device. The VERSION
# decides the tier verdict, never --source-quant: dsl-fp8 quantizes a BF16
# checkpoint at load time and still moves 8-bit bytes (the same rule the roof
# byte model below uses). dsl-q8 (05 Q1) and dsl-w4 (05 Q2) are the plan's arms
# A2/A3/A4: the seam accepts them now so a future quantizer needs no harness
# change, and the tier gate already knows their capacity class. The engine
# rejects a --version it has not implemented at launch, which is the fail-closed
# direction (no silent substitution of another format).
# Which CHECKPOINT each version loads. Every variant defaults to the BF16
# checkpoint except dsl-q8, which only exists for a Q8_0 checkpoint (the CLI
# refuses a dsl-q8 run on a non-Q8_0 file, and refuses a Q8_0 file for every
# other version, so a mismatched --model fails closed instead of silently
# substituting a format).
VERSION_DEFAULT_MODEL = {
    "dsl-q8": "data/Qwen3.5-4B-Q8_0.gguf",
}
DEFAULT_MODEL = "data/Qwen3.5-4B-BF16.gguf"


def version_default_model(version: str) -> str:
    """Checkpoint a version loads when --model is not given."""
    return VERSION_DEFAULT_MODEL.get(version, DEFAULT_MODEL)


def resolve_arm_model(version: str, model_arg: str | None) -> str:
    """The --model value for one (arm, executable) invocation.

    `--model` wins; without it the version's own checkpoint is used, so
    `--versions dsl-q8` needs no extra flag. Mixing a Q8_0 checkpoint with a
    non-q8 version (or the reverse) is a CLI-level failure; the harness never
    rewrites the checkpoint to make a version "work".
    """
    if model_arg:
        return model_arg
    return version_default_model(version)


VERSION_WEIGHT_FORMAT = {
    "dsl-bf16": "bf16",
    "dsl-fp8": "fp8",
    "tile-bf16": "bf16",
    "tile-fp8": "fp8",
    "dsl-q8": "q8_0",
    "dsl-w4": "w4",
}

# Decode byte model of 08 section 2.1/2.2 + 03 section 4.1. It is a DOCUMENTED
# model, not a measurement: the weight bytes are the plan's per-format totals and
# the KV term is the 8 full-attention layers x 2 x 4 kv heads x 256 head_dim x
# 4 B FP32 per token, times the KV READ amplification of the current decode
# attention kernel. It exists only so a "roof %" can be computed against a
# MEASURED --bandwidth; an unknown source_quant yields n/a instead of a guess.
# Keyed by the ENGINE's weight storage (the `storage=` field of the CLI's
# `engine:` line: bf16_packed / fp8_packed / f32_dense), because that is what the
# decode loop actually streams. source_quant describes the checkpoint and is the
# wrong input for an arm whose weights are quantized at load time (a
# dsl-fp8 arm on a BF16 checkpoint moves FP8 bytes, not BF16 bytes). The plain
# quant names stay as aliases for summaries that predate the storage field.
DOCUMENTED_WEIGHT_BYTES = {
    "bf16_packed": 8.41e9,
    "fp8_packed": 4.48e9,
# Native Q8_0 (phase 8a / 05 Q1): 1.0625 B/weight of int8 block codes plus
 # one FP32 scale per 32 weights per row, i.e. 1.1875 B/weight on the device.
 # Measured real-checkpoint allocation is 4514.92 MiB = 4.73424e9 B; using the
 # source GGUF's 1.0625 B/weight here would omit the FP32 scale widening.
    "q8_0_packed": 4.73424e9,
      # Phase 8b W4A16 decode (plan option (c)). What ONE decode token streams:
      # 0.5 B/weight of int4 codes + 4 B per 128-weight group = 0.03125 B/weight
      # for every W4 matrix, plus the LM HEAD, which stays 8-bit by contract
      # (248320 x 2560 x 1.0625 B = 0.675e9 B). Derivation from the measured Q8_0
      # allocation (4.73424e9 B at 1.0625 B/weight, LM head included):
      #   (4.73424e9 - 0.675e9) * (0.53125 / 1.0625) + 0.675e9 = 2.705e9 B.
      # The FP8 PREFILL copy is not read by a decode step (its traffic shows up in
      # TTFT, not here), which is exactly why option (c) costs memory, not tokens.
      "w4_packed": 2.705e9,
    "bf16": 8.41e9,
    "f16": 8.41e9,
    "fp8": 4.48e9,
          "q8_0": 4.73424e9,
      "w4": 2.705e9,
      "w4_packed": 2.705e9,
      "int8": 4.47e9,
}
# 03 section 4.1/4.2: the legacy make_qwen_attn_decode_kernel runs ONE BLOCK PER
# q_head and every one of the 4 q_heads of a KV head streams the same K/V rows
# (kvh = h / (q_heads / kv_heads)), i.e. the cache is read 4x per attended
# position. The split-KV + GQA-dedup family (03 section 4.2) stages K/V once per
# (kv_head, key_split) for all 4 q_heads and therefore reads exactly 1x.
#
# Since Phase 5 the engine REPORTS which of the two ran, so this is no longer a
# global constant: `decode_attn_dedup_applies()` decides per arm from the engine's
# own `decode attn:` line plus the context the measured step actually ran at. The
# 4x constant below stays as the LEGACY model (and the default when nothing is
# reported), which is what every stored report before Phase 5 was scored with --
# flipping it globally would silently rewrite those numbers.
DECODE_KV_READ_AMPLIFICATION = 4 # per-q_head decode kernel (03 section 4.1)
DECODE_KV_READ_AMPLIFICATION_DEDUP = 1  # split-KV + GQA dedup (03 section 4.2)
ROOF_MODEL_NOTE = (
    "roof % = achieved decode bytes/token divided by the MEASURED --bandwidth; "
    "bytes/token = weight bytes per ENGINE weight storage (08 section 2.1: "
          "bf16_packed 8.41e9, fp8_packed 4.48e9, q8_0 4.47e9, w4_packed 2.705e9 = "
      "int4 codes + one FP32 scale per 128 K + the 8-bit LM head) + FP32 KV 65536 B/token "
    "at the observed context (08 section 2.2) times the KV read amplification of "
    "the decode attention that ACTUALLY ran. That amplification is 4x for the "
    "legacy per-q_head kernel (03 section 4.1: each of the 4 q_heads of a KV head "
    "re-reads the same K/V rows) and 1x for the split-KV + GQA-dedup family "
    "(03 section 4.2). The harness only claims the 1x term when the engine's own "
    "`decode attn:` line reports active=split AND the measured step's context is "
    "at or above that engine's reported threshold; otherwise -- legacy engine, "
    "split disabled, short context, or no decode-attn line at all -- it keeps the "
    "4x term. It is a model, not a counter: n/a without --bandwidth or for an "
    "unknown storage")


def decode_attn_dedup_applies(engine: dict | None, observed_ctx: int) -> bool:
    """Whether the measured decode step read the KV cache once (03 section 4.2).

    True ONLY when the engine reported the split family active AND the step's kv
    length reached that engine's threshold. `observed_ctx` is the context the
    measured decode steps ran at (prefill tokens + decoded steps), i.e. the same
    number the KV term is computed from -- a run that never crossed the threshold
    keeps the 4x legacy term even on a dedup-capable engine.
    """
    attn = (engine or {}).get("decode_attn") if engine else None
    if not attn or attn.get("active") != "split":
        return False
    threshold = int(attn.get("threshold") or 0)
    if threshold < 1:
        return False
    return int(observed_ctx) >= threshold


def weight_format_for_version(version: str | None) -> str | None:
    """Weight format class of an engine version (08 section 5), or None.

    None means "this harness does not know the format". It is NOT a licence to
    run any length: tier_format_max_context() then fails closed.
    """
    if not version:
        return None
    return VERSION_WEIGHT_FORMAT.get(version.strip())


def tier_capacity_gib(tier: str, weight_format: str | None,
                         tokens: int, kv_dtype: str | None = "f32") -> dict | None:
    """08 section 5's VRAM budget for one (tier, format, context) pair.

    Returns the component rows of that section for `tokens` tokens of context,
    or None for a format the harness does not know (nothing to add up).
    """
    weights = FORMAT_WEIGHT_GIB.get(weight_format or "")
    if weights is None:
        return None
    kv = max(int(tokens), 0) * kv_bytes_per_token(kv_dtype) / GIB
    return {
        "tier_usable_gib": TIER_USABLE_GIB[tier],
        "weight_gib": weights,
        "kv_gib": kv,
        "fixed_gib": FIXED_OVERHEAD_GIB,
        "total_gib": weights + kv + FIXED_OVERHEAD_GIB,
  }


def tier_format_max_context(tier: str, weight_format: str | None,
                            kv_dtype: str | None = "f32") -> int:
    """Largest context (tokens) 08 section 5 says the tier holds in a format.

    The usable VRAM minus the format's weight bytes minus the 0.15 GiB fixed
    term is the room for FP32 KV at 64 KiB/token, which is exactly the 0.25 /
    2.00 / 8.00 GiB that section lists for ctx 4 K / 32 K / 128 K. The result is
    clamped to the model's own 262144 limit, and an UNKNOWN format returns the
    length every format holds (fail closed) instead of the tier's maximum.
    """
    budget = tier_capacity_gib(tier, weight_format, 0, kv_dtype)
    if budget is None:
        return FORMAT_UNKNOWN_MAX_CONTEXT[tier]
    available = (budget["tier_usable_gib"] - budget["weight_gib"]
                 - FIXED_OVERHEAD_GIB)
    if available <= 0.0:
        return 0
    return min(int(available * GIB // kv_bytes_per_token(kv_dtype)), MODEL_MAX_CONTEXT)


# The tier ceiling across formats (the summary stamp and the report header): the
# longest context ANY format can hold, i.e. the tier's real ceiling rather than
# the BF16 one. The per-version ceilings are what the gate uses.
#
# Deliberate consequence, stated so no one reads it as a bug: because 08 section 5
# publishes verdicts for specific lengths, solving its budget is a touch more
# permissive than its summary sentence where the section is SILENT (BF16 on T16
# between 32 K and the 115015-token arithmetic ceiling fits: 7.83 + 4.00 + 0.15 =
# 11.98 GiB at 64 K). Every configuration the section calls "does not fit" is
# still refused (BF16 128 K and 262144, FP8/Q8_0/W4 262144), and the two lengths
# it blesses per format (BF16 32768 / 128 K at 8 bits or less, 262144 on T32) are
# covered by tests/test_benchmark_qwen.py.
TIER_MAX_CONTEXT = {
    tier: max(tier_format_max_context(tier, fmt) for fmt in FORMAT_WEIGHT_GIB)
    for tier in TIERS
}

# GPU sampling contract (08 section 7 step 8, 08 section 9 H3). One nvidia-smi
# call returns all of these, so the throttle probe costs no extra process.
# Defaults are the plan's "is anyone else using the box?" gate: >5% GPU
# utilization before launch means a foreign tenant, and a recorded but not
# enforced memory.used ceiling (0 = disabled) keeps desktop machines usable.
DEFAULT_GPU_IDLE_UTIL_PCT = 5.0
DEFAULT_GPU_IDLE_MAX_USED_MIB = 0

# source_quant values this harness understands; anything else is still accepted
# (a future quantizer) but recorded verbatim so arms stay distinguishable.
KNOWN_SOURCE_QUANTS = ("bf16", "f16", "q8_0", "fp8", "int8")
# A value outside that tuple is still accepted and recorded verbatim (a future
# quantizer must not need a harness change), but it is marked unverified.

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
# Provenance lines of example_qwen: the engine summary printed right after
# engine creation (cache/workspace bytes, compiled shader count) and the final
# counter line (cuda-graph steps == recorded graph launches, updates ==
# re-records). Parsed so every number in the summary JSON can be traced to a
# log line instead of a hand-typed value.
_ENGINE_RE = re.compile(
    r"^engine: kernel=(\S+) storage=(\S+) max_context=(\d+) prefill_chunk=(\d+) "
    r"gdn=(\S+) (?:prefill_attn=\S+ )?cache=(\d+(?:\.\d+)?) MiB "
    r"workspace=(\d+(?:\.\d+)?) MiB "
    r"shaders=(\d+) \(compile (\d+(?:\.\d+)?) ms\)[ \t]*$",
    re.MULTILINE,
)
_ENGINE_FINAL_RE = re.compile(
    # `native-prefill-launches` was appended to this line in phase 7a; it is
    # optional here so logs from every phase still parse.
    r"^engine final: position=(\d+) shaders=(\d+) cuda-graph-steps=(\d+) "
    r"updates=(\d+)(?: native-prefill-launches=(\d+))?[ \t]*$",
    re.MULTILINE,
)


# Phase 3 provenance (opt_plan/02 sections 3+4, 09 V6/N3.5). Optional: logs from
# earlier phases do not contain them, but whenever they are present they are
# parsed, de-duplicated and range-checked so a summary can never claim a kernel
# or a chunk that the engine did not actually use.
_PREFILL_GEMM_RE = re.compile(r"^prefill gemm: (mt\(mt=(\d+)\)|legacy)[ \t]*$",
                              re.MULTILINE)
_WORKSPACE_RE = re.compile(
    r"^workspace: total=(\d+(?:\.\d+)?) MiB chunk=(\d+(?:\.\d+)?) MiB "
    r"pool_per_row=(\d+) floats[ \t]*$",
    re.MULTILINE,
)
# Phase 4 (opt_plan/03 section 2.3, M-c): which decode schedule the run asked for
# and which one the engine could host. Optional like the block below, so logs from
# earlier phases still parse; `active == "static"` is what makes a
# `cuda_graph_updates == 0` counter meaningful.
_DECODE_GRAPH_RE = re.compile(
    r"^decode graph: requested=(static|off) active=(static|off|unavailable) "
    r"cuda-graph-ext=(\d+) storage=(\S+)[ \t]*$",
    re.MULTILINE,
)


# Phase 5 (opt_plan/03 section 4.2, M-c): which decode attention realization the
# engine compiled and where its legacy/split threshold sits. Optional like the
# blocks above (logs from earlier phases still parse). `active == "split"` plus
# `threshold` is exactly what the roof model needs: it may only claim the 1x KV
# term for a step whose kv_len reached the threshold, and never for a run whose
# engine kept the per-q_head kernel.
_DECODE_ATTN_RE = re.compile(
    r"^decode attn: requested=(split|legacy) active=(split|legacy|unavailable) "
    r"threshold=(\d+) max_splits=(\d+) k_tile=(\d+) shared=(\d+) "
    r"partial_bytes=(\d+) blocks=(\d+) splits=(\d+) kv_len=(\d+) dedup=(\d) "
    r"sm=(\d+) sm_source=(\S+)[ \t]*$",
    re.MULTILINE,
)


_PREFILL_CHUNKING_RE = re.compile(
    r"^prefill chunking: mode=(adaptive|fixed) configured_chunk=(\d+) "
    r"configured_max=(\d+) backend_limit=(\d+) selected=(\d+) chunks=(\d+)"
    r"[ \t]*$",
    re.MULTILINE,
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


def parse_engine_lines(stdout: str, label: str) -> dict:
    """Extract the engine provenance of ONE example_qwen run.

    Returns kernel/storage/max_context/prefill_chunk/gdn/cache+workspace bytes,
    the compiled shader count, the engine-compile time and the final cuda-graph
    step/update counters. Fail-closed like the rest of the parser: a missing or
    duplicated line aborts instead of leaving a summary without provenance.
    max_context here is what the BINARY decided it can hold, which is exactly
    what --context-lengths must be checked against.
    """
    stdout = _normalize_newlines(stdout)
    engine = _ENGINE_RE.findall(stdout)
    if len(engine) != 1:
        raise BenchmarkError(
            f"{label}: expected exactly one 'engine: ...' line, found "
            f"{len(engine)}; unsupported/malformed CLI output")
    (kernel, storage, max_context, prefill_chunk, gdn, cache_mib,
     workspace_mib, shaders, compile_ms) = engine[0]
    final = _ENGINE_FINAL_RE.findall(stdout)
    if len(final) != 1:
        raise BenchmarkError(
            f"{label}: expected exactly one 'engine final: ...' line, found "
            f"{len(final)}; unsupported/malformed CLI output")
    position, final_shaders, graph_steps, graph_updates, _native_prefill = final[0]
    if int(shaders) != int(final_shaders):
        raise BenchmarkError(
            f"{label}: compiled shader count changed between the engine line "
            f"({shaders}) and the final line ({final_shaders}); the run is not "
            f"the one that was measured")

    # Phase 3 provenance (optional, validated when present): the prefill
    # projection kernel, the workspace split and -- most importantly -- the
    # chunking the engine REALLY selected for this prompt (configured vs used).
    gemm = _PREFILL_GEMM_RE.findall(stdout)
    if len(gemm) > 1:
        raise BenchmarkError(
            f"{label}: found {len(gemm)} 'prefill gemm: ...' lines; expected at "
            f"most one")
    workspace = _WORKSPACE_RE.findall(stdout)
    if len(workspace) > 1:
        raise BenchmarkError(
            f"{label}: found {len(workspace)} 'workspace: ...' lines; expected at "
            f"most one")
    graph_matches = _DECODE_GRAPH_RE.findall(stdout)
    if len(graph_matches) > 1:
        raise BenchmarkError(
            f"{label}: found {len(graph_matches)} 'decode graph: ...' lines; "
            f"expected at most one")
    chunking = _PREFILL_CHUNKING_RE.findall(stdout)
    if len(chunking) > 1:
        raise BenchmarkError(
            f"{label}: found {len(chunking)} 'prefill chunking: ...' lines; "
            f"expected at most one")
    chunk_info = None
    if chunking:
        (mode, configured_chunk, configured_max, backend_limit, selected,
         chunks) = chunking[0]
        configured_chunk = int(configured_chunk)
        configured_max = int(configured_max)
        backend_limit = int(backend_limit)
        selected = int(selected)
        chunks = int(chunks)
        if selected < 1 or selected > max(configured_chunk, configured_max):
            raise BenchmarkError(
                f"{label}: 'prefill chunking' selected={selected} exceeds the "
                f"configured chunk/max ({configured_chunk}/{configured_max})")
        if backend_limit and selected > backend_limit:
            raise BenchmarkError(
                f"{label}: 'prefill chunking' selected={selected} exceeds the "
                f"backend limit {backend_limit}")
        if mode == "fixed" and selected != configured_chunk:
            raise BenchmarkError(
                f"{label}: 'prefill chunking' mode=fixed but selected="
                f"{selected} != configured_chunk={configured_chunk}")
        if chunks < 1:
            raise BenchmarkError(
                f"{label}: 'prefill chunking' reports chunks={chunks}")

        chunk_info = {
            "mode": mode,
            "configured_chunk": configured_chunk,
            "configured_max": configured_max,
            "backend_limit": backend_limit,
            "selected": selected,
            "chunks": chunks,
        }


    attn_matches = _DECODE_ATTN_RE.findall(stdout)
    if len(attn_matches) > 1:
        raise BenchmarkError(
            f"{label}: found {len(attn_matches)} 'decode attn: ...' lines; "
            f"expected at most one")
    attn_info = None
    if attn_matches:
        (requested, active, threshold, max_splits, k_tile, shared_bytes,
         partial_bytes, blocks, splits, kv_len, dedup, sm_count,
         sm_source) = attn_matches[0]
        threshold = int(threshold)
        max_splits = int(max_splits)
        k_tile = int(k_tile)
        blocks = int(blocks)
        splits = int(splits)
        kv_len = int(kv_len)
        dedup = int(dedup)
        sm_count = int(sm_count)
        if dedup not in (0, 1):
            raise BenchmarkError(f"{label}: 'decode attn' reports dedup={dedup}")
        if active == "split":
            if threshold < 1:
                raise BenchmarkError(
                    f"{label}: 'decode attn' reports active=split with "
                    f"threshold={threshold}; the threshold must be >= 1")
            if splits > max_splits:
                raise BenchmarkError(
                    f"{label}: 'decode attn' reports splits={splits} > "
                    f"max_splits={max_splits}")
            # The engine's dedup flag must agree with its own threshold rule: a step
            # below the threshold cannot have used the split family.
            if dedup and kv_len < threshold:
                raise BenchmarkError(
                    f"{label}: 'decode attn' reports dedup=1 at kv_len={kv_len} "
                    f"below the split threshold {threshold}")
        elif dedup:
            raise BenchmarkError(
                f"{label}: 'decode attn' reports dedup=1 while active={active}; "
                f"only the split family dedups the KV read")
        attn_info = {
            "requested": requested,
            "active": active,
            "threshold": threshold,
            "max_splits": max_splits,
            "k_tile": k_tile,
            "shared_bytes": int(shared_bytes),
            "partial_bytes": int(partial_bytes),
            "blocks": blocks,
            "splits": splits,
            "kv_len": kv_len,
            "dedup": bool(dedup),
            # A backend that answers no SM-count query (dx/vk) reports 0: it must
            # survive as 0 so the report says "blocks/SM unavailable" instead of
            # dividing by a guess. cuda answers the real count (see
            # patches/luisa-cuda-multiprocessor-query.patch), and then the report
            # divides by a MEASURED value.
            "sm_count": sm_count,
            "sm_source": sm_source,
        }
    graph_info = None
    if graph_matches:
        requested, active, has_ext, storage_arm = graph_matches[0]
        if has_ext not in ("0", "1"):
            raise BenchmarkError(
                f"{label}: 'decode graph' reports cuda-graph-ext={has_ext}")
        graph_info = {
            "requested": requested,
            "active": active,
            "cuda_graph_ext": has_ext == "1",
            "storage": storage_arm,
        }
    return {
        "kernel": kernel,
        "weight_storage": storage,
        "reported_max_context": int(max_context),
        "reported_prefill_chunk": int(prefill_chunk),
        "gdn": gdn,
        "cache_mib": float(cache_mib),
        "workspace_mib": float(workspace_mib),
        "compiled_shaders": int(shaders),
        "engine_compile_ms_cli": float(compile_ms),
        "final_position": int(position),
        "cuda_graph_steps": int(graph_steps),
        "cuda_graph_updates": int(graph_updates),
        "prefill_gemm": (gemm[0][0] if gemm else None),
        "prefill_gemm_mt": (int(gemm[0][1]) if gemm and gemm[0][1] else 0),
        "workspace_total_mib": (float(workspace[0][0]) if workspace else None),
        "workspace_chunk_mib": (float(workspace[0][1]) if workspace else None),
        "workspace_pool_floats_per_row": (
            int(workspace[0][2]) if workspace else None),
        "prefill_chunking": chunk_info,
        "decode_graph": graph_info,

        "decode_attn": attn_info,
  }


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
# Deterministic synthetic prompts (--prompt-lengths / --context-lengths)
# ---------------------------------------------------------------------------

# Rule (repeated in the docstring and in --prompt-lengths' help so it can be
# audited from three places): a 64-bit LCG with explicit integer arithmetic --
# no floats, no hash()/set iteration order, no dependence on the random
# module's implementation -- seeded from the requested length. The same length
# therefore yields identical ids on every run, every machine and every Python
# version.
_PROMPT_SEED = 0x0C0FFEE5EEDBEEF01
_PROMPT_LENGTH_MIX = 0xD1B54A32D192ED03
_PROMPT_LCG_MUL = 6364136223846793005  # Knuth/PCG LCG multiplier
_PROMPT_LCG_INC = 1442695040888963407  # Knuth/PCG LCG increment
_U64_MASK = 0xFFFFFFFFFFFFFFFF

# Qwen3.5-4B reports vocab 248320 (docs/examples/example_qwen.md). Ids are
# drawn from [PROMPT_ID_MIN, PROMPT_ID_MAX), a range that excludes the low
# control ids and the high special-token block; example_qwen rejects
# out-of-vocab ids loudly ("prompt token <i> has id <id>, outside the loaded
# vocabulary"), so a mismatched checkpoint fails instead of silently
# benchmarking another prompt.
PROMPT_ID_MIN = 1000
PROMPT_ID_MAX = 200000
QWEN_VOCAB_SIZE = 248320


def synthetic_prompt_ids(length: int) -> list[int]:
    """Deterministic token-id list of exactly `length` ids.

    Rule:
        state = (PROMPT_SEED ^ (length * PROMPT_LENGTH_MIX)) & (2**64 - 1)
        repeat `length` times:
            state = (state * A + C) & (2**64 - 1)      # A, C = PCG LCG constants
            id    = PROMPT_ID_MIN
                    + ((state >> 32) % (PROMPT_ID_MAX - PROMPT_ID_MIN))

    Pure integer arithmetic, so the output is identical on any machine. Ids are
    always inside [PROMPT_ID_MIN, PROMPT_ID_MAX) and therefore inside the
    Qwen3.5-4B vocabulary.
    """
    if length < 1:
        raise BenchmarkError(f"prompt length must be >= 1, got {length}")
    span = PROMPT_ID_MAX - PROMPT_ID_MIN
    state = (_PROMPT_SEED ^ ((length * _PROMPT_LENGTH_MIX) & _U64_MASK)) & _U64_MASK
    ids = []
    for _ in range(length):
        state = (state * _PROMPT_LCG_MUL + _PROMPT_LCG_INC) & _U64_MASK
        ids.append(PROMPT_ID_MIN + (state >> 32) % span)
    return ids


def ids_to_arg(ids: list[int]) -> str:
    """The CLI's --tokens spelling: comma separated, no spaces."""
    return ",".join(str(i) for i in ids)


def parse_token_arg(tokens: str) -> list[int]:
    """Parse --tokens strictly; a bad list is a fatal argument error."""
    try:
        ids = [int(t) for t in tokens.split(",")]
    except ValueError:
        raise BenchmarkError("--tokens must be a comma-separated integer list")
    if not ids or any(t < 0 for t in ids):
        raise BenchmarkError("--tokens must be non-empty non-negative ids")
    return ids


def parse_length_list(text: str | None, flag: str) -> list[int]:
    """Parse a --prompt-lengths / --context-lengths list; fail closed."""
    if text is None:
        return []
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise BenchmarkError(f"{flag} contains an empty entry")
        try:
            value = int(item)
        except ValueError:
            raise BenchmarkError(
                f"{flag} must be comma-separated positive integers, got {item!r}")
        if value < 1:
            raise BenchmarkError(f"{flag} values must be >= 1, got {value}")
        values.append(value)
    if len(set(values)) != len(values):
        raise BenchmarkError(f"{flag} contains duplicate values")
    return values


def arm_key(prompt_length: int, context_length: int) -> str:
    """Attribution key shared by plan labels, statistics buckets, the id
    reference and the markdown report. It records the prompt length, so a
    number can never be attributed to the wrong prefill length."""
    return f"P{prompt_length}-C{context_length}"


def _synthetic_arm(prompt_length: int, context_length: int) -> dict:
    return {
        "key": arm_key(prompt_length, context_length),
        "prompt_length": prompt_length,
        "context_length": context_length,
        "tokens": ids_to_arg(synthetic_prompt_ids(prompt_length)),
        # The ids are exactly what --synthetic-prompt-len regenerates, so the
        # CLI can rebuild them when the id list does not fit a command line.
        "synthetic": True,
        "origin": "synthetic",
  }


def plan_sweep(args) -> tuple[list[dict], int, bool]:
    """Expand the CLI into arms plus the single --max-context to pass.

    One arm is one (prompt_length, context_length) measurement unit and one
    statistics/attribution bucket:
      * neither flag: the legacy single arm from --tokens / --max-context;
      * only --prompt-lengths: one arm per P; the context bucket is the KV
        capacity that will actually be passed;
      * only --context-lengths: the prompt length of arm C is C, so the decode
        loop runs at ctx ~= C and TPOT is attributed to C;
      * both: cross product, C is the KV capacity and P the prefill length.

    --max-context is always the largest requested context plus the decode
    budget, so a sweep needs no hand-tuned KV capacity. Returns
    (arms, max_context, sweeping).
    """
    plens = parse_length_list(args.prompt_lengths, "--prompt-lengths")
    clens = parse_length_list(args.context_lengths, "--context-lengths")
    n_new = args.max_new_tokens
    if not plens and not clens:
        ids = parse_token_arg(args.tokens)
        if len(ids) + n_new > args.max_context:
            raise BenchmarkError(
                f"prompt ({len(ids)}) + max-new-tokens ({n_new}) exceeds "
                f"max-context ({args.max_context})")
        arms = [{
            "key": arm_key(len(ids), args.max_context),
            "prompt_length": len(ids),
            "context_length": args.max_context,
            "tokens": args.tokens,
            # An explicit id list is NOT synthetic_prompt_ids(P) in general, so
            # --synthetic-prompt-len cannot stand in for it; a long one has to
            # travel as --tokens-file.
            "synthetic": ids == synthetic_prompt_ids(len(ids)),
            "origin": "--tokens",
        }]
        return arms, args.max_context, False
    pairs: list[tuple[int, int]]
    if plens and clens:
        for p in plens:
            for c in clens:
                if p + n_new > c:
                    raise BenchmarkError(
                        f"--prompt-lengths {p} with --context-lengths {c}: the "
                        f"prompt ({p}) plus --max-new-tokens ({n_new}) does not "
                        f"fit in the requested context length {c}")
        capacity = max([args.max_context] + clens + [max(plens) + n_new])
        pairs = [(p, c) for p in plens for c in clens]
    elif plens:
        capacity = max([args.max_context] + [p + n_new for p in plens])
        pairs = [(p, capacity) for p in plens]
    else:
        capacity = max([args.max_context] + clens + [c + n_new for c in clens])
        pairs = [(c, c) for c in clens]
    arms = [_synthetic_arm(p, c) for p, c in pairs]
    if len({a["key"] for a in arms}) != len(arms):
        raise BenchmarkError("internal: duplicate arm keys in the sweep")
    return arms, capacity, True


def resolve_repeat(args, sweeping: bool) -> int:
    """--repeat resolution. A single run is never reported (plan risk B4)."""
    requested = getattr(args, "repeat", None)
    if requested is None:
        return SWEEP_REPEAT_DEFAULT if sweeping else DEFAULT_REPEAT
    if requested < MIN_REPEAT:
        raise BenchmarkError(
            f"--repeat must be >= {MIN_REPEAT}: a single run is never reported "
            f"(median/p95 need a distribution)")
    if sweeping and requested < SWEEP_REPEAT_DEFAULT:
        raise BenchmarkError(
            f"--repeat must be >= {SWEEP_REPEAT_DEFAULT} for a sweep "
            f"(--prompt-lengths/--context-lengths); got {requested}")
    return requested


def resolve_source_quant(flag_value: str | None, model: str) -> tuple[str, str]:
    """Mandatory source_quant + where the value came from.

    The origin is recorded so a reader can tell a declared quantization apart
    from one inferred from the checkpoint's file name (plan risk B3: a
    double-quantized Q8_0-sourced arm must never be silently mixed with a
    BF16-sourced FP8 arm).
    """
    if flag_value:
        value = flag_value.strip()
        if not value:
            raise BenchmarkError("--source-quant must not be empty")
        origin = "flag" if value in KNOWN_SOURCE_QUANTS else "flag-unverified"
        return value, origin
    name = Path(model).name.upper()
    for marker, value in (("Q8_0", "q8_0"), ("BF16", "bf16"), ("F16", "f16"),
                          ("FP8", "fp8")):
        if marker in name:
            return value, "inferred-from-model-filename"
    return "unspecified", "unspecified"


# ---------------------------------------------------------------------------
# Tiers, measured-bandwidth roof and the prompt transport
# ---------------------------------------------------------------------------

def resolve_tier(tier: str | None) -> str | None:
    """Validate --tier; None means "not declared" (a run may omit it, a report
    may not)."""
    if tier is None:
        return None
    value = tier.strip().lower()
    if value not in TIERS:
        raise BenchmarkError(
            f"--tier must be one of {', '.join(TIERS)}; got {tier!r}")
    return value


def resolve_device_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    value = tag.strip()
    if not value:
        raise BenchmarkError("--device-tag must not be empty")
    return value


def resolve_bandwidth(value: float | None) -> float | None:
    """--bandwidth is a MEASURED achievable DRAM bandwidth in decimal GB/s."""
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkError(
            f"--bandwidth must be a positive measured DRAM bandwidth in GB/s; "
            f"got {value!r}")
    return float(value)


def check_tier_limits(tier: str | None, arms: list[dict], *,
                      max_new_tokens: int,
                      versions: list[str] | tuple[str, ...] | None = None,
                      max_context: int | None = None,
                      kv_dtype: str | None = "f32") -> None:
    """Refuse an arm the (tier, weight format) pair cannot hold (08 section 5).

    Capacity is per FORMAT, not per tier: 7.83 GiB of BF16 weights do not fit
    next to a 128 K FP32 KV inside T16's 15.0 GiB (16.0 GiB total), while FP8 /
    Q8_0 (12.3 GiB) and W4 (10.2 GiB) do, so T16 reaches 128 K only at 8 bits or
    less; T32 holds the model's native 262144 in every format. Both halves of the
    requirement are checked, because either one can blow the budget: the prompt
    plus the decode budget, and the --max-context actually passed (a 128-token
    prompt with --max-context 262144 still allocates 262144 tokens of KV).

    A version whose weight format this harness does not know fails closed to the
    length every format holds instead of running ungated. Without --tier nothing
    is tier-gated (the summary then records "tier unknown" and the report refuses
    to render), but the model's own 262144 limit always applies.
    """
    for arm in arms:
        needed = arm["prompt_length"] + max_new_tokens
        capacity = max([needed, int(arm["context_length"])]
                       + ([int(max_context)] if max_context is not None else []))
        if needed > MODEL_MAX_CONTEXT or capacity > MODEL_MAX_CONTEXT:
            raise BenchmarkError(
                f"arm {arm['key']}: prompt {arm['prompt_length']} + "
                f"--max-new-tokens {max_new_tokens} = {needed} and the context "
                f"actually passed ({capacity}) must fit the model's "
                f"{MODEL_MAX_CONTEXT}-token context")
        if tier is None:
            continue
        for version in (versions or (None,)):
            fmt = weight_format_for_version(version)
            cap = tier_format_max_context(tier, fmt, kv_dtype)
            if capacity <= cap:
                continue
            label = (f"{version} / {fmt if fmt else 'unknown format'} weights"
                     if version else
                     f"{fmt if fmt else 'unknown format'} weights")
            budget = tier_capacity_gib(tier, fmt, capacity, kv_dtype)
            fits = (f"{budget['total_gib']:.2f} GiB = "
                    f"{budget['weight_gib']:.2f} GiB of weights + "
                    f"{budget['kv_gib']:.2f} GiB of {kv_dtype or 'f32'} KV at {capacity} tokens "
                    f"+ {budget['fixed_gib']:.2f} GiB fixed, against "
                    f"{budget['tier_usable_gib']:.1f} GiB usable"
                    if budget else
                    f"{capacity} tokens against a ceiling of {cap} tokens")
            if fmt is None:
                hint = ("This version's weight format is unknown to the "
                        "harness, which fails closed to 32768 tokens on BOTH "
                        "tiers: add the format to FORMAT_WEIGHT_GIB (and the "
                        "version to VERSION_WEIGHT_FORMAT) once 05 defines its "
                        "storage size, or drop the length.")
            elif tier == "t16" and fmt == "bf16":
                hint = (f"8-bit-or-less weights reach 128 K on T16 "
                        f"(08 section 5): rerun with --versions dsl-fp8, dsl-q8 "
                        f"or dsl-w4 (arms A1-A3), drop the length, or use "
                        f"--tier t32.")
            else:
                hint = ("Drop the length or run it with --tier t32.")
            raise BenchmarkError(
                f"arm {arm['key']} / {label}: prompt {arm['prompt_length']} + "
                f"--max-new-tokens {max_new_tokens} = {needed}, and the context "
                f"actually passed ({capacity}) does not fit tier {tier}, whose "
                f"{fmt if fmt else 'unknown-format'} context ceiling is {cap} "
                f"tokens ({fits} by 08 section 5). " + hint)


def decode_kv_read_amplification(*, kv_dedup: bool | None = None) -> int:
    """How many times a decode step reads the KV cache (03 section 4.1/4.2).

    `kv_dedup=None` keeps the legacy 4x term, which is also the documented model
    for an engine that does not report its decode attention (every stored report
    before Phase 5) -- the constant is NOT flipped, so old numbers keep their
    meaning. Callers that parse the engine's `decode attn:` line pass
    `decode_attn_dedup_applies(engine, observed_ctx)`, which is True only when the
    split family ran for the context that was measured.
    """
    if kv_dedup is None:
        return DECODE_KV_READ_AMPLIFICATION
    return (DECODE_KV_READ_AMPLIFICATION_DEDUP if kv_dedup
            else DECODE_KV_READ_AMPLIFICATION)


def kv_bytes_read_per_token(observed_ctx: int, *,
                            kv_dedup: bool | None = None,
                            kv_dtype: str | None = "f32") -> float:
    """FP32 KV bytes ONE decode step reads at `observed_ctx` attended positions.

    Unique bytes per token x the KV read amplification of the kernel that ran, so
    a per-q_head decode gets a 4x larger KV term (03 section 4.1) and the split-KV
    + GQA-dedup family 1x (03 section 4.2).
    """
    return (float(kv_bytes_per_token(kv_dtype)) * max(int(observed_ctx), 0)
            * decode_kv_read_amplification(kv_dedup=kv_dedup))


def roof_bytes_per_token(weight_storage: str, observed_ctx: int, *,
                         kv_dedup: bool | None = None,
                         kv_dtype: str | None = "f32") -> float | None:
    """Documented decoder byte model (08 section 2.1/2.2 + 03 section 4.1), or None.

    `weight_storage` is the ENGINE's device storage (the CLI's `engine:` line
    reports bf16_packed / fp8_packed / f32_dense), not the checkpoint's
    source_quant: a dsl-fp8 arm quantizes a BF16 checkpoint at load time and
    streams FP8 bytes.

    Weights are read once per token. The KV term is 64 KiB of FP32 KV per token of
    context, wrapped in the CURRENT kernel's read amplification
    (decode_kv_read_amplification(): 4x today, because the per-q_head decode
    attention re-reads every K/V row for all four q_heads of a KV head; 03
    section 4.1), so it grows with the observed context and the model never claims
    a dedup that has not landed. The model is stated in the summary/report next to
    the number it produced, and an unknown storage yields None -> the roof column
    prints n/a.
    """
    weights = DOCUMENTED_WEIGHT_BYTES.get(weight_storage)
    if weights is None:
        return None
    return weights + kv_bytes_read_per_token(observed_ctx, kv_dedup=kv_dedup,
                                             kv_dtype=kv_dtype)


def roof_percent(tokens_per_s: float | None, bytes_per_token: float | None,
                 bandwidth_gbps: float | None) -> float | None:
    """Achieved decode bandwidth as a percentage of the MEASURED roof.

    None (printed as n/a) when any input is missing: never a vendor number.
    """
    if tokens_per_s is None or bytes_per_token is None or not bandwidth_gbps:
        return None
    achieved = tokens_per_s * bytes_per_token            # bytes/s
    return achieved / (bandwidth_gbps * 1e9) * 100.0


def achieved_gbps(tokens_per_s: float | None,
                  bytes_per_token: float | None) -> float | None:
    if tokens_per_s is None or bytes_per_token is None:
        return None
    return tokens_per_s * bytes_per_token / 1e9


def prompt_transport(arm: dict, *, base_cmd: list[str],
                     max_chars: int) -> tuple[str, object]:
    """How this arm's ids reach example_qwen.

    Order: --tokens when the whole command line fits, else
    --synthetic-prompt-len when the arm's ids ARE synthetic_prompt_ids(P) (the
    CLI regenerates the same list from the same constants), else --tokens-file.
    Returns (kind, value) with kind in {"tokens", "synthetic", "file"}.
    """
    candidate = base_cmd + ["--tokens", arm["tokens"]]
    if projected_command_chars(candidate) <= max_chars:
        return "tokens", arm["tokens"]
    if arm.get("synthetic"):
        return "synthetic", arm["prompt_length"]
    return "file", None


def file_fingerprint(path: Path) -> dict:
    """Size + mtime of an input file, so a number can be traced to a binary,
    a checkpoint or a reference file that changed (or did not)."""
    try:
        st = path.stat()
    except OSError as e:
        return {"path": str(path), "exists": False, "size_bytes": None,
                "mtime_utc": None, "error": str(e)}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(
            st.st_mtime, timezone.utc).isoformat(),
  }


def projected_command_chars(cmd: list[str]) -> int:
    """Upper bound of the command-line length the OS will see.

    Windows quoting is the worst case, so it is used on every platform: the
    number is then a safe upper bound for POSIX too.
    """
    return len(subprocess.list2cmdline(cmd))


def command_length_error(cmd: list[str], *, max_chars: int, label: str) -> str | None:
    """Return an explanation when the command line cannot be launched.

    Since example_qwen accepts --synthetic-prompt-len and --tokens-file this is
    only a final guard: prompt_transport() has already moved the ids off the
    command line when they did not fit.
    """
    total = projected_command_chars(cmd)
    if total <= max_chars:
        return None
    return (
        f"{label}: the synthesized command line is {total} characters, above "
        f"the {max_chars}-character limit checked by --max-argv-chars. Windows "
        f"caps a whole command line at 32767 characters; example_qwen accepts "
        f"--synthetic-prompt-len and --tokens-file precisely so a long prompt "
        f"never has to travel here, so this means the fallback itself failed. "
        f"Lower the prompt/context length, raise --max-argv-chars if the "
        f"platform really allows it, or run the same arm on Linux (2 MiB "
        f"limit). Nothing was launched.")


# ---------------------------------------------------------------------------
# Host provenance: GPU identity + coarse VRAM sampling
# ---------------------------------------------------------------------------

def _run_capture(cmd: list[str], *, timeout: float = 20.0) -> str | None:
    """Run a short read-only helper (nvidia-smi). None on any failure, so a
    machine without a GPU keeps working instead of aborting."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _query_gpu_rows(fields: str, *, capture=None) -> list[list[str]] | None:
    """CSV rows of one `nvidia-smi --query-gpu` call, or None."""
    run = _run_capture if capture is None else capture
    out = run(["nvidia-smi", f"--query-gpu={fields}",
               "--format=csv,noheader,nounits"])
    if out is None:
        return None
    rows = []
    for line in _normalize_newlines(out).splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows or None


def _cell_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def query_gpu_memory(*, gpu_index: int = 0, capture=None) -> dict | None:
    """memory.used/free/total of one GPU in MiB, or None (no nvidia-smi).

    Records the nvidia-smi index it read; CUDA_VISIBLE_DEVICES is NOT resolved
    by this harness (there is no device-ordering guarantee), which the summary
    states explicitly instead of pretending it knows the device under test.
    """
    rows = _query_gpu_rows("index,memory.used,memory.free,memory.total",
                           capture=capture)
    if not rows:
        return None
    for row in rows:
        if len(row) < 4:
            continue
        if _cell_int(row[0]) == gpu_index:
            return {
                "gpu_index": gpu_index,
                "gpu_count": len(rows),
                "used_mib": _cell_int(row[1]),
                "free_mib": _cell_int(row[2]),
                "total_mib": _cell_int(row[3]),
            }
    return None


def query_gpu_identity(*, gpu_index: int = 0, capture=None) -> dict:
    """Name / driver / total memory / compute capability of one GPU.

    Every field is null when nvidia-smi is missing or does not support the
    query; the run continues (the harness must stay usable without a GPU).
    """
    identity = {
        "nvidia_smi_available": False,
        "gpu_index": gpu_index,
        "name": None,
        "driver_version": None,
        "memory_total_mib": None,
        "compute_capability": None,
  }
    rows = _query_gpu_rows("index,name,driver_version,memory.total",
                           capture=capture)
    if not rows:
        return identity
    for row in rows:
        if len(row) >= 4 and _cell_int(row[0]) == gpu_index:
            identity.update({
                "nvidia_smi_available": True,
                "name": row[1],
                "driver_version": row[2],
                "memory_total_mib": _cell_int(row[3]),
            })
            break
    # compute_cap needs a recent driver; a failure here must not null the rest.
    cc_rows = _query_gpu_rows("index,compute_cap", capture=capture)
    if cc_rows:
        for row in cc_rows:
            if len(row) >= 2 and _cell_int(row[0]) == gpu_index:
                identity["compute_capability"] = row[1]
                break
    return identity


def query_gpu_used_mib(*, gpu_index: int = 0, capture=None) -> int | None:
    """One `--query-gpu=memory.used` sample in MiB (the plan's exact command),
    or None when nvidia-smi is unavailable."""
    rows = _query_gpu_rows("index,memory.used", capture=capture)
    if not rows:
        return None
    for row in rows:
        if len(row) >= 2 and _cell_int(row[0]) == gpu_index:
            return _cell_int(row[1])
    return None


# The quantities 08 section 7 step 8 asks for (plus memory.total, which turns
# the sample into a free-VRAM pre-flight). One nvidia-smi invocation returns all
# of them, so the throttle probe adds no process per sample.
GPU_SAMPLE_FIELDS = ("memory.used", "memory.total", "utilization.gpu",
                     "clocks.sm", "power.draw", "temperature.gpu")


def _cell_float(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def query_gpu_state(*, gpu_index: int = 0, capture=None) -> dict | None:
    """One full GPU sample (memory, utilization, SM clock, power, temperature).

    Every individual cell may be `[N/A]` (laptops without a power sensor,
    virtualized clocks, ...) or missing from a short row; such a field is null
    and the rest of the sample survives. None means nvidia-smi itself is
    unavailable, which the callers treat as "unmeasured", never as zero.
    """
    rows = _query_gpu_rows("index," + ",".join(GPU_SAMPLE_FIELDS),
                           capture=capture)
    if not rows:
        return None
    for row in rows:
        if not row or _cell_int(row[0]) != gpu_index:
            continue

        def cell(i: int, parse):
            return parse(row[i]) if i < len(row) else None

        return {
            "gpu_index": gpu_index,
            "fields": ["index", *GPU_SAMPLE_FIELDS],
            "used_mib": cell(1, _cell_int),
            "total_mib": cell(2, _cell_int),
            "utilization_pct": cell(3, _cell_int),
            "sm_clock_mhz": cell(4, _cell_int),
            "power_w": cell(5, _cell_float),
            # nvidia-smi reports temperature.gpu as a float ("27.83"), not an
            # integer, on current drivers; _cell_int would silently null it.
            "temperature_c": cell(6, _cell_float),
        }
    return None


def gpu_state_flat(state: dict | None) -> dict:
    """The throttle-relevant fields of a sample, without the raw cells."""
    if not state:
        return {"available": False, "used_mib": None, "utilization_pct": None,
                "sm_clock_mhz": None, "power_w": None, "temperature_c": None}
    return {"available": True,
            "used_mib": state.get("used_mib"),
            "utilization_pct": state.get("utilization_pct"),
            "sm_clock_mhz": state.get("sm_clock_mhz"),
            "power_w": state.get("power_w"),
            "temperature_c": state.get("temperature_c")}


class VramPoller:
    """Coarse GPU sampler around ONE benchmark process.

    Polls `nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,
    power.draw,temperature.gpu --format=csv,noheader,nounits` every `interval_s`
    from a daemon thread while the process runs. Coarse by construction -- each
    sample is a separate nvidia-smi process launch -- so the reported peak is a
    SAMPLED maximum, never an instantaneous one, and the post-prefill instant
    cannot be isolated. The clock/power/temperature cells are the throttle probe
    of 08 section 7 step 8: a sustained 5-15 % clock drop is visible in the
    samples instead of implied. Never raises: without nvidia-smi it records
    nothing and `available` stays False.
    """

    def __init__(self, interval_s: float = DEFAULT_VRAM_INTERVAL_S, *,
                 gpu_index: int = 0, capture=None,
                 clock=time.monotonic) -> None:
        self.interval_s = max(0.02, float(interval_s))
        self.gpu_index = gpu_index
        self._capture = capture
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict] = []
        self._t0 = 0.0
        self.available: bool | None = None

    def _sample_once(self) -> None:
        state = query_gpu_state(gpu_index=self.gpu_index,
                                capture=self._capture)
        if state is None or state.get("used_mib") is None:
            self.available = False
            return
        self.available = True
        self._samples.append({"t_s": round(self._clock() - self._t0, 3),
                              "used_mib": state["used_mib"],
                              "utilization_pct": state["utilization_pct"],
                              "sm_clock_mhz": state["sm_clock_mhz"],
                              "power_w": state["power_w"],
                              "temperature_c": state["temperature_c"]})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def start(self) -> None:
        self._t0 = self._clock()
        self._samples = []
        self._stop.clear()
        self._sample_once()
        if self.available:
            self._thread = threading.Thread(
                target=self._loop, name="qwen-vram-poll", daemon=True)
            self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 4))
            self._thread = None
        return list(self._samples)


def summarize_throttle(*, before_state: dict | None, after_state: dict | None,
                       during_samples: list[dict]) -> dict:
    """Before/after throttle probe of 08 section 7 step 8 plus the extremes of
    the in-flight samples. Every value is a coarse nvidia-smi sample; a missing
    sensor stays null instead of being reported as zero."""
    clocks = [s.get("sm_clock_mhz") for s in during_samples
              if s.get("sm_clock_mhz") is not None]
    powers = [s.get("power_w") for s in during_samples
              if s.get("power_w") is not None]
    temps = [s.get("temperature_c") for s in during_samples
             if s.get("temperature_c") is not None]
    return {
        "note": ("nvidia-smi samples: clocks.sm/power.draw/temperature.gpu "
                 "before and after the run, plus the sampled extremes while the "
                 "process ran. sm_clock_change_mhz is before minus after: a "
                 "POSITIVE value is a clock drop. An idle before-sample reads "
                 "the idle down-clock (hundreds of MHz), so compare `after` "
                 "against a warm run or use min_sm_clock_mhz for the sustained "
                 "throttling of 08 section 9 H4 -- never a code regression."),
        "before": gpu_state_flat(before_state),
        "after": gpu_state_flat(after_state),
        "during_sample_count": len(during_samples),
        # min_sm_clock_mhz is polluted by the idle down-clock at process start
        # (a 210 MHz sample is normal before the driver ramps), so the median of
        # the in-flight samples is the number to read for sustained throttling.
        "min_sm_clock_mhz": min(clocks) if clocks else None,
        "median_sm_clock_mhz": (statistics.median(clocks) if clocks else None),
        "max_power_w": max(powers) if powers else None,
        "max_temperature_c": max(temps) if temps else None,
        "sm_clock_change_mhz": (
            before_state["sm_clock_mhz"] - after_state["sm_clock_mhz"]
            if before_state and after_state and
            before_state.get("sm_clock_mhz") is not None and
            after_state.get("sm_clock_mhz") is not None else None),
  }


def summarize_vram_samples(*, before_start_mib: int | None,
                           during_samples: list[dict],
                           after_exit_mib: int | None,
                           interval_s: float,
                           before_state: dict | None = None,
                           after_state: dict | None = None) -> dict:
    """Map the coarse samples onto the plan's three checkpoints, honestly.

    The plan asks for before-start / after-load+prefill / after-decode. A
    poller cannot see the prefill instant, so:
      * before_start_mib -- the one sample taken before launch;
      * after_load_prefill_mib -- the max of the in-flight samples, labelled a
        coarse UPPER BOUND of the load+prefill+decode resident set, because the
        prefill instant is not observable from outside the process;
      * after_decode_mib -- the last in-flight sample (decode is the last
        phase before the process exits);
      * peak_used_mib -- max of everything, the number the report uses.

    `before_state`/`after_state` are the full GPU samples (memory, utilization,
    clock, power, temperature) that feed the throttle probe; passing them never
    changes the VRAM numbers above.
    """
    used = [s["used_mib"] for s in during_samples if s["used_mib"] is not None]
    for value in (before_start_mib, after_exit_mib):
        if value is not None:
            used.append(value)
    utils = [s.get("utilization_pct") for s in during_samples
             if s.get("utilization_pct") is not None]
    return {
        "enabled": True,
        "sampling": ("coarse poll of nvidia-smi memory.used, utilization.gpu, "
                     "clocks.sm, power.draw, temperature.gpu"),
        "sample_interval_s": interval_s,
        "labels": {
            "before_start_mib": "sampled before the benchmark process existed",
            "after_load_prefill_mib":
                "COARSE UPPER BOUND of the in-flight resident set; nvidia-smi "
                "polling cannot isolate the instant after prefill",
            "after_decode_mib":
                "last in-flight sample (decode is the last phase before exit); "
                "null when the process was shorter than one poll interval",
            "peak_used_mib": "sampled maximum over every sample above",
        },
        "sample_count": len(during_samples),
        "before_start_mib": before_start_mib,
        "after_load_prefill_mib": max(used) if used else None,
        "after_decode_mib": (during_samples[-1]["used_mib"]
                             if during_samples else after_exit_mib),
        "after_exit_mib": after_exit_mib,
        "peak_used_mib": max(used) if used else None,
        "peak_utilization_pct": max(utils) if utils else None,
        "throttle": summarize_throttle(before_state=before_state,
                                       after_state=after_state,
                                       during_samples=during_samples),
        "samples": during_samples,
  }



def preflight_vram(free_mib: int | None, *, min_free_mib: int,
                   gpu: dict | None = None) -> dict:
    """Pre-launch VRAM decision (pure). Never guesses a number it lacks."""
    gpu = gpu or {}
    name = gpu.get("name")
    label = f"GPU {gpu.get('gpu_index', 0)}" + (f" ({name})" if name else "")
    if min_free_mib < 0:
        raise BenchmarkError("--min-free-vram-mib must be >= 0")
    if free_mib is None:
        return {
            "status": "unavailable",
            "free_mib": None,
            "min_free_mib": min_free_mib,
            "message": ("nvidia-smi is unavailable: free-VRAM pre-flight "
                        "skipped and recorded as null (the run continues)"),
        }
    if free_mib < min_free_mib:
        return {
            "status": "low",
            "free_mib": free_mib,
            "min_free_mib": min_free_mib,
            "message": (
                f"ABORTING before launch: free VRAM {free_mib} MiB on {label} "
                f"is below --min-free-vram-mib {min_free_mib}. Nothing has "
                f"been launched. Free the GPU (close other CUDA/D3D/Vulkan "
                f"clients, check `nvidia-smi --query-gpu=memory.free "
                f"--format=csv`) and re-run; a timing run on a busy desktop GPU "
                f"is not a measurement."),
        }
    return {
        "status": "ok",
        "free_mib": free_mib,
        "min_free_mib": min_free_mib,
        "message": (f"free VRAM {free_mib} MiB on {label} >= "
                    f"--min-free-vram-mib {min_free_mib}"),
  }


def preflight_gpu_idle(state: dict | None, *, min_free_mib: int,
                       max_util_pct: float = DEFAULT_GPU_IDLE_UTIL_PCT,
                       max_used_mib: int = DEFAULT_GPU_IDLE_MAX_USED_MIB,
                       enforced: bool = True,
                       gpu: dict | None = None) -> dict:
    """Pre-launch "is anyone else using this GPU?" decision (pure).

    08 section 9 H3: a shared box (another tenant, remote-desktop streaming, a
    leftover CUDA process) makes a timing run meaningless, and a "regression" is
    then just another process. The check uses the same single nvidia-smi sample
    as the free-VRAM pre-flight:
      * free VRAM below --min-free-vram-mib      -> refused (status "low");
      * utilization.gpu above --gpu-idle-util-pct -> refused (status "busy")
        unless --no-idle-check downgrades it to a recorded warning;
      * memory.used above --gpu-idle-max-used-mib (when > 0) -> refused.
    A missing nvidia-smi is "unavailable": recorded, never a refusal, so the
    harness keeps working on a machine without a GPU.
    """
    gpu = gpu or {}
    name = gpu.get("name")
    label = f"GPU {gpu.get('gpu_index', 0)}" + (f" ({name})" if name else "")
    if min_free_mib < 0:
        raise BenchmarkError("--min-free-vram-mib must be >= 0")
    if max_util_pct < 0:
        raise BenchmarkError("--gpu-idle-util-pct must be >= 0")
    if max_used_mib < 0:
        raise BenchmarkError("--gpu-idle-max-used-mib must be >= 0")
    if state is None:
        return {
            "status": "unavailable",
            "enforced": enforced,
            "utilization_pct": None,
            "used_mib": None,
            "free_mib": None,
            "message": ("nvidia-smi is unavailable: the idle/foreign-activity "
                        "pre-flight is skipped and recorded as null (the run "
                        "continues)"),
        }
    used = state.get("used_mib")
    util = state.get("utilization_pct")
    # Prefer an explicit free_mib (query_gpu_memory's answer); a full sample
    # only carries total/used, so derive it there.
    if state.get("free_mib") is not None:
        free = state["free_mib"]
    elif state.get("total_mib") is not None and used is not None:
        free = state["total_mib"] - used
    else:
        free = None
    base = {
        "enforced": enforced,
        "gpu_label": label,
        "utilization_pct": util,
        "used_mib": used,
        "free_mib": free,
        "max_util_pct": max_util_pct,
        "max_used_mib": max_used_mib,
        "min_free_mib": min_free_mib,
        "throttle": summarize_throttle(before_state=state, after_state=None,
                                       during_samples=[]),
  }
    if free is not None and free < min_free_mib:
        return {**base, "status": "low", "message": (
            f"ABORTING before launch: free VRAM {free} MiB on {label} is below "
            f"--min-free-vram-mib {min_free_mib}. Nothing has been launched. "
            f"Free the GPU (close other CUDA/D3D/Vulkan clients) and re-run; a "
            f"timing run on a busy desktop GPU is not a measurement.")}
    busy_reasons = []
    if util is not None and util > max_util_pct:
        busy_reasons.append(f"utilization.gpu {util}% > --gpu-idle-util-pct "
                            f"{max_util_pct}%")
    if max_used_mib > 0 and used is not None and used > max_used_mib:
        busy_reasons.append(f"memory.used {used} MiB > --gpu-idle-max-used-mib "
                            f"{max_used_mib}")
    if busy_reasons and enforced:
        return {**base, "status": "busy", "message": (
            f"ABORTING before launch: foreign GPU activity on {label} ("
            + "; ".join(busy_reasons)
            + "). Nothing has been launched. 08 section 9 H3: a shared box is "
              "not a measurement -- close the other client (or stop the remote "
              "session) and re-run, or pass --no-idle-check to record the "
              "warning and continue.")}
    if busy_reasons:
        return {**base, "status": "warning", "message": (
            f"foreign GPU activity on {label} (" + "; ".join(busy_reasons)
            + ") recorded as a WARNING because --no-idle-check is set; treat "
              "the timings as indicative, not as a measurement")}
    return {**base, "status": "ok", "message": (
        f"idle pre-flight ok on {label}: utilization "
        f"{'n/a' if util is None else str(util) + '%'}, memory.used "
        f"{'n/a' if used is None else str(used) + ' MiB'}")}


# ---------------------------------------------------------------------------
# ID references (--write-id-reference / --require-id-match)
# ---------------------------------------------------------------------------

def build_id_reference(*, backend: str, model: str, source_quant: str,
                       max_new_tokens: int, arms: list[dict]) -> dict:
    """Build the reference document for the schema documented in the module
    docstring. `arms` are the finished arm records of a run."""
    doc = {
        "schema": ID_REFERENCE_SCHEMA,
        "version": ID_REFERENCE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model": model,
        "source_quant": source_quant,
        "max_new_tokens": max_new_tokens,
        "note": ("Greedy generated ids per arm, keyed by "
                 "P<prompt_length>-C<context_length>. Compare with "
                 "--require-id-match; an arm's ids are what proves an "
                 "optimization did not change the model."),
        "arms": {},
  }
    for arm in arms:
        ids = arm.get("generated_ids")
        if ids is None:
            raise BenchmarkError(
                f"arm {arm['arm']}: no ids to store in the reference")
        doc["arms"][arm["arm"]] = {
            "prompt_length": arm["prompt_length"],
            "context_length": arm["context_length"],
            "tokens": max_new_tokens,
            "ids": list(ids),
            "stop": arm.get("stop"),
            "engine_version": arm["version"],
        }
    return doc


def write_id_reference(path: Path, reference: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")


def load_id_reference(path: Path) -> dict:
    """Load and structurally validate a reference; fail closed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise BenchmarkError(f"--require-id-match: cannot read {path}: {e}")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise BenchmarkError(f"--require-id-match: {path} is not JSON: {e}")
    if not isinstance(doc, dict):
        raise BenchmarkError(f"--require-id-match: {path} is not a JSON object")
    schema = doc.get("schema")
    if schema is not None and schema != ID_REFERENCE_SCHEMA:
        raise BenchmarkError(
            f"--require-id-match: {path} declares schema {schema!r}, expected "
            f"{ID_REFERENCE_SCHEMA!r}")
    version = doc.get("version")
    if version != ID_REFERENCE_VERSION:
        raise BenchmarkError(
            f"--require-id-match: {path} declares version {version!r}, this "
            f"harness speaks version {ID_REFERENCE_VERSION}")
    arms = doc.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise BenchmarkError(
            f"--require-id-match: {path} has no 'arms' object; the reference "
            f"schema is documented in scripts/benchmark_qwen.py")
    for key, entry in arms.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("ids"), list):
            raise BenchmarkError(
                f"--require-id-match: {path}: arm {key} has no 'ids' list")
    return doc


def compare_ids_to_reference(reference: dict, *, arm: str,
                             ids: list[int]) -> dict:
    """Hard-fail when one arm's ids differ from the stored reference."""
    entry = reference["arms"].get(arm)
    if entry is None:
        raise BenchmarkError(
            f"arm {arm}: not present in the reference "
            f"({', '.join(sorted(reference['arms']))}); refusing to pass an arm "
            f"that was never fingerprinted")
    ref_ids = [int(i) for i in entry["ids"]]
    if ref_ids == list(ids):
        return {"arm": arm, "match": True, "reference_len": len(ref_ids),
                "first_difference_index": None}
    limit = min(len(ref_ids), len(ids))
    index = next((i for i in range(limit) if ref_ids[i] != ids[i]), limit)
    ref_token = ref_ids[index] if index < len(ref_ids) else None
    run_token = ids[index] if index < len(ids) else None
    raise BenchmarkError(
        f"arm {arm}: generated ids differ from the reference at index {index} "
        f"(reference {ref_token} vs run {run_token}; reference has "
        f"{len(ref_ids)} ids, the run generated {len(ids)}). The optimization "
        f"changed the model output -- that is a hard failure, not a tuning "
        f"result (plan gate 2).")


# ---------------------------------------------------------------------------
# Process execution (strictly sequential)
# ---------------------------------------------------------------------------

def build_command(exe: str, version: str, args, csv_path: Path | None, *,
                  prompt: tuple[str, object], max_context: int) -> list[str]:
    """One example_qwen invocation. `prompt` is the arm's transport
    ("tokens", ids) / ("synthetic", length) / ("file", path) and `max_context`
    the sweep's KV capacity, so a sweep never accidentally reuses the CLI's
    global --tokens."""
    kind, value = prompt
    cmd = [
        exe,
        "--backend", args.backend,
        "--version", version,
        "--model", resolve_arm_model(version, args.model),
    ]
    if getattr(args, "kv_dtype", None):
        cmd += ["--kv-dtype", args.kv_dtype]
    if kind == "tokens":
        cmd += ["--tokens", str(value)]
    elif kind == "synthetic":
        cmd += ["--synthetic-prompt-len", str(value)]
    elif kind == "file":
        cmd += ["--tokens-file", str(value)]
    elif kind == "none":
        pass                                  # base form, prompt_transport() probe
    else:                                     # pragma: no cover - internal bug
        raise BenchmarkError(f"internal: unknown prompt transport {kind!r}")
    cmd += [
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-context", str(max_context),
        "--headless",
    ]
    if getattr(args, "decode_flight", None) is not None:
        cmd += ["--decode-flight", str(args.decode_flight)]
    # opt_plan/09 N4.2-N4.4 (phase 9b): conditional WHILE chunk, pinned download
    # staging and the PDL creation request. None means "leave the binary default"
    # (the WHILE is off and pinned staging is on by default), so an arm that does not
    # ask for them keeps the exact historical command line.
    if getattr(args, "cuda_graph_while", None) is not None:
        cmd += ["--cuda-graph-while", str(args.cuda_graph_while)]
    if getattr(args, "graph_flight", False):
        cmd += ["--graph-flight"]
    if getattr(args, "pinned_transfers", None) is not None:
        cmd += ["--pinned-transfers", "1" if args.pinned_transfers else "0"]
    if getattr(args, "pdl", None) is not None:
        cmd += ["--pdl", "1" if args.pdl else "0"]
    if csv_path is not None:
        cmd += ["--benchmark-csv", str(csv_path)]
    # W4 weights are read from an offline pack only (never quantized inside a
    # request). The flag is passed to every version so the CLI can fail closed
    # on a mismatched combination instead of the harness guessing.
    w4_pack = getattr(args, "w4_pack", None)
    if w4_pack and version == "dsl-w4":
        cmd += ["--w4-pack", str(w4_pack)]
    if version == "dsl-w4" and getattr(args, "w4_prefill_in_kernel", False):
        cmd += ["--w4-prefill", "in-kernel"]
    # OFFLINE merged input projections (opt_plan/03 section 3 + D5). Passed to
    # every non-W4 version when requested, so the CLI (not the harness) decides
    # whether the pack's format matches the variant; --merged-decode 0 is the A/B
    # arm that keeps the per-projection GEMVs on the pack's segment views.
    pack = getattr(args, "projection_pack", None)
    if pack and version != "dsl-w4":
        cmd += ["--projection-pack", str(pack)]
        if getattr(args, "merged_decode", True) is False:
            cmd += ["--merged-decode", "0"]
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


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy "linear" / R type 7).

    Documented because the harness must be reproducible across machines: with
    5 repetitions p95 is an interpolation between the two largest observations,
    NOT an observed repetition, and the summary/markdown say so.
    """
    if not values:
        raise BenchmarkError("percentile of an empty sample set")
    if not 0.0 <= pct <= 100.0:
        raise BenchmarkError(f"percentile out of range: {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stat_triple(values: list[float]) -> dict:
    """median / min / p95 / max of one metric. Never a single run: --repeat is
    enforced elsewhere (resolve_repeat)."""
    return {"median": statistics.median(values),
            "min": min(values), "max": max(values),
            "p95": percentile(values, 95.0)}


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


def group_samples_by_arm(samples: list[dict]) -> dict[tuple, dict[str, list]]:
    """Bucket samples by (version, prompt_length, context_length), then by
    executable. The arm is the unit of statistics: two prompt lengths or two
    context buckets must never share a median."""
    grouped: dict[tuple, dict[str, list]] = {}
    for sample in samples:
        key = (sample["version"], sample["prompt_length"],
               sample["context_length"])
        grouped.setdefault(key, {"new": [], "baseline": []})[
            sample["which"]].append(sample)
    return grouped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ids_match_text(value) -> str:
    if value is None:
        return "not-checked"
    return "yes" if value else "no"


def _arms_by_version(summary: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for arm in summary.get("arms", []):
        grouped.setdefault(arm["version"], []).append(arm)
    return grouped


def _arm_vram_line(vram: dict | None) -> str:
    if not vram:
        return "- VRAM: not sampled (--vram was not given)"
    return (
        f"- VRAM (COARSE poll every {vram['sample_interval_s']}s; "
        f"{vram['sample_count']} in-flight samples): before start "
        f"{_fmt(vram['before_start_mib'], 0)} MiB | after load/prefill "
        f"{_fmt(vram['after_load_prefill_mib'], 0)} MiB (upper bound, the "
        f"prefill instant is not observable) | after decode "
        f"{_fmt(vram['after_decode_mib'], 0)} MiB | after exit "
        f"{_fmt(vram['after_exit_mib'], 0)} MiB | sampled peak "
        f"{_fmt(vram['peak_used_mib'], 0)} MiB")


def _arm_roof_line(roof: dict) -> str:
    """The per-arm decode roof bullet: the number AND the model behind it.

    The KV read amplification is repeated here because it is a property of the
    decode attention that RAN (03 section 4.1 vs 4.2), not of the model: a reader
    must be able to see which one produced the percentage. Summaries written
    before Phase 5 have no `kv_dedup` key; they render with the legacy wording so
    an old report keeps saying exactly what it was scored with.
    """
    amplification = roof.get("kv_read_amplification",
                             DECODE_KV_READ_AMPLIFICATION)
    if "kv_dedup" in roof:
        if roof.get("kv_dedup"):
            claim = ("the engine's split-KV + GQA-dedup path ran, 03 section 4.2 "
                     f"(threshold {roof.get('kv_split_threshold')})" )
        elif roof.get("kv_dedup_reported_by_engine"):
            claim = ("the engine has split-KV but this context stayed below its "
                     f"threshold {roof.get('kv_split_threshold')}, so the legacy "
                     "per-q_head kernel ran, 03 section 4.1")
        elif roof.get("kv_dedup_reported_by_engine") is None:
            claim = ("the engine did not report its decode attention, so the "
                     "legacy per-q_head kernel is assumed, 03 section 4.1")
        else:
            claim = ("the engine kept the legacy per-q_head kernel, "
                     "03 section 4.1")
    else:
        dedup = roof.get("kv_dedup_reported_by_engine")
        claim = ("the engine reports KV dedup, 03 section 4.2" if dedup
                 else "no dedup reported by the engine, 03 section 4.1")
    return (
        f"- decode roof: {_fmt(roof.get('achieved_gbps'), 1)} GB/s achieved of "
        f"{_fmt(roof.get('bandwidth_gbps'), 1)} GB/s measured = "
        f"{_fmt(roof.get('roof_pct'), 1)} % at ctx "
        f"{roof.get('observed_context_tokens')} "
        f"({_fmt(roof.get('bytes_per_token'), 0)} B/token, documented model: "
        f"KV read {amplification}x per attended position, "
        f"{claim})")


def _arm_engine_line(engine: dict | None) -> str:
    if not engine:
        return "- engine: n/a"
    chunk = engine.get("prefill_chunking") or {}
    gemm = engine.get("prefill_gemm")
    tail = ""
    if gemm:
        tail += f" prefill_gemm={gemm}"
    if chunk:
        tail += (f" prefill_chunk_mode={chunk['mode']}"
                 f" prefill_chunk_used={chunk['selected']}/{chunk['chunks']}chunks")
    if engine.get("workspace_chunk_mib") is not None:
        tail += f" workspace_chunk={_fmt(engine['workspace_chunk_mib'], 2)} MiB"
    return (
        f"- engine: kernel={engine['kernel']} storage="
        f"{engine['weight_storage']} max_context={engine['reported_max_context']} "
        f"prefill_chunk={engine['reported_prefill_chunk']} gdn={engine['gdn']} "
        f"shaders={engine['compiled_shaders']} (compile "
        f"{_fmt(engine['engine_compile_ms_cli'], 1)} ms) cache="
        f"{_fmt(engine['cache_mib'], 2)} MiB workspace="
        f"{_fmt(engine['workspace_mib'], 2)} MiB cuda-graph-steps="
        f"{engine['cuda_graph_steps']} updates={engine['cuda_graph_updates']}" + tail)



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
    a(f"- Tier: `{summary.get('tier') or 'UNDECLARED'}` | device tag: "
      f"`{summary.get('device_tag') or 'UNDECLARED'}` | measured bandwidth: "
      f"{_fmt(summary.get('bandwidth_gbps'), 1)} GB/s (roof denominator)")
    a(f"- source_quant: `{ctx['source_quant']}` ({ctx['source_quant_origin']}) "
      f"| ttft_kind: `{ctx['ttft_kind']}`")
    a(f"- Arms: {', '.join(ctx['arm_labels']) or 'none'} | max-new-tokens: "
      f"{ctx['max_new_tokens']} | max-context passed: {ctx['max_context']} | "
      f"repeat: {ctx['repeat']} | warmup: {ctx.get('warmup', 0)} "
      f"(excluded from statistics, correctness-checked)")
    if ctx.get("ttft_semantics"):
        a(f"- TTFT semantics ({ctx['ttft_kind']}): {ctx['ttft_semantics']}")
    a(f"- Host: {ctx['platform']} | Python {ctx['python']}")
    prov = summary.get("provenance") or {}
    gpu = prov.get("gpu") or {}
    if gpu:
        a(f"- GPU: {gpu.get('name')} | compute capability "
          f"{gpu.get('compute_capability')} | driver "
          f"{gpu.get('driver_version')} | nvidia-smi "
          f"{'available' if gpu.get('nvidia_smi_available') else 'not available'}")
    pre = prov.get("vram_preflight") or {}
    a(f"- Free VRAM before start: {_fmt(pre.get('free_mib'), 0)} MiB "
      f"(--min-free-vram-mib {pre.get('min_free_mib', 'n/a')}, "
      f"{pre.get('status', 'unknown')})")
    idle = prov.get("idle_preflight") or {}
    if idle:
        a(f"- Idle pre-flight: {idle.get('status', 'unknown')} "
          f"(utilization {_fmt(idle.get('utilization_pct'), 0)}%, memory.used "
          f"{_fmt(idle.get('used_mib'), 0)} MiB, "
          f"{'enforced' if idle.get('enforced') else 'recording only'})")
    thr = prov.get("throttle_probe") or {}
    if thr.get("before") or thr.get("after"):
        before, after = thr.get("before", {}), thr.get("after", {})
        a(f"- Throttle probe: sm clock {_fmt(before.get('sm_clock_mhz'), 0)} -> "
          f"{_fmt(after.get('sm_clock_mhz'), 0)} MHz | power "
          f"{_fmt(before.get('power_w'), 1)} -> {_fmt(after.get('power_w'), 1)} W "
          f"| temperature {_fmt(before.get('temperature_c'), 0)} -> "
          f"{_fmt(after.get('temperature_c'), 0)} C"
          + (f" | sampled median clock "
             f"{_fmt(thr.get('median_sm_clock_mhz'), 0)} MHz (min "
             f"{_fmt(thr.get('min_sm_clock_mhz'), 0)} incl. the idle ramp), "
             f"max power {_fmt(thr.get('max_power_w'), 1)} W, max temperature "
             f"{_fmt(thr.get('max_temperature_c'), 0)} C"
             if thr.get('during_sample_count') else ""))
    a(f"- Roof model: {summary.get('roof_model', ROOF_MODEL_NOTE)}")
    for key, label in (("exe", "Executable file"), ("model", "Model file")):
        fp = prov.get(key) or {}
        if fp:
            a(f"- {label}: `{fp.get('path')}` size {fp.get('size_bytes')} bytes, "
              f"mtime {fp.get('mtime_utc')}")
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
    a("p95 uses linear interpolation (percentile()); with few repetitions it "
      "is an interpolation between the two largest observations, not an "
      "observed run.")
    a("")
    for version, arms in _arms_by_version(summary).items():
        a(f"### {version}")
        a("")
        for arm in arms:
            a(f"#### {arm['arm']}")
            a("")
            a(f"- prompt length {arm['prompt_length']} | context bucket "
              f"{arm['context_length']} | max-context passed "
              f"{arm['max_context']} | ttft_kind `{arm['ttft_kind']}` | "
              f"source_quant `{arm['source_quant']}` | repeats "
              f"{arm['repeat']} | ids match: {_ids_match_text(arm.get('ids_match'))}")
            a(_arm_engine_line(arm.get("engine")))
            roof = arm.get("roof") or {}
            if roof:
                a(_arm_roof_line(roof))
            a(_arm_vram_line(arm.get("vram")))
            a("")
            a(f"Generated ids (identical across all repeats): "
              f"`{arm['generated_ids']}` (stop: {arm['stop']})")
            a("")
            for which, stats in arm["stats"].items():
                a(f"**{which}** (n={stats['n']})")
                a("")
                a("| metric | median | min | p95 | max |")
                a("|---|---|---|---|---|")
                for key in _METRIC_KEYS:
                    s = stats[key]
                    if s is None:
                        a(f"| {key} | n/a | n/a | n/a | n/a |")
                    else:
                        a(f"| {key} | {_fmt(s['median'])} | {_fmt(s['min'])} | "
                          f"{_fmt(s['p95'])} | {_fmt(s['max'])} |")
                a("")
            if "baseline_comparison" in arm:
                a("**Paired baseline comparison** (new vs baseline, same "
                  "repetition; interleaved baseline,new / new,baseline). "
                  "Percent change = (new - baseline) / baseline * 100; for time "
                  "metrics a NEGATIVE change is an improvement.")
                a("")
                a("| metric | median delta | median % change | min % | max % |")
                a("|---|---|---|---|")
                for key in _METRIC_KEYS:
                    s = arm["baseline_comparison"]["summary"][key]
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
      "process wall time, engine line and raw log paths. All logs/CSVs are "
      f"preserved in `{summary['run_dir']}`.")
    a("")
    path.write_text("\n".join(lines), encoding="utf-8")


# Columns required for docs/examples/example_qwen.md. Every value is read from
# the collected summary JSON -- the report is generated, never retyped.
REPORT_COLUMNS = (
    "version / arm", "exe", "backend", "model", "quant source",
    "tier / device", "max_context", "prompt length", "TTFT ms", "ms/token",
    "tokens/s", "decode GB/s", "roof %", "peak VRAM MiB", "repeats",
    "ids match",
)


def build_report_rows(summary: dict) -> list[dict]:
    """One row per (arm, executable) built from the summary dict alone.

    tier/device_tag/bandwidth are stamped here too, so a copied row can never
    be attributed to the other tier and the roof column can never be recomputed
    from a number that is not in the summary.
    """
    rows = []
    bandwidth = summary.get("bandwidth_gbps")
    # Stored KV dtype of the whole run (opt_plan/05 Q3); a bf16 cache halves the
    # KV term of the documented byte model, which is the point of the dtype arm.
    summary_kv_dtype = summary.get("kv_dtype")
    for arm in summary.get("arms", []):
        for which in ("new", "baseline"):
            stats = arm.get("stats", {}).get(which)
            if stats is None:
                continue
            ttft = stats["ttft_ms"]
            tpot = stats["decode_ms_per_token"]
            tps = stats["decode_tokens_per_s"]
            vram = arm.get("vram") or {}
            tokens_per_s = tps["median"] if tps else None
            roof = arm.get("roof") or {}
            bytes_per_token = roof.get("bytes_per_token")
            if "bytes_per_token" not in roof:
                # A summary that predates the roof block still renders: the byte
                # model is recomputed from the engine's storage line.
                observed_ctx = (arm.get("context_attribution", {}) or {}).get(
                    "observed_decode_context_tokens")
                if observed_ctx is None:
                    observed_ctx = arm["prompt_length"]
                bytes_per_token = roof_bytes_per_token(
                    (arm.get("engine") or {}).get("weight_storage")
                    or arm["source_quant"], int(observed_ctx))
            rows.append({
                "version": arm["version"],
                "arm": arm["arm"],
                "which": which,
                "backend": arm["backend"],
                "model": Path(arm["model"]).name,
                "source_quant": arm["source_quant"],
                "tier": summary.get("tier"),
                "device_tag": summary.get("device_tag"),
                "bandwidth_gbps": bandwidth,
                "max_context": arm["max_context"],
                "prompt_length": arm["prompt_length"],
                "context_length": arm["context_length"],
                "ttft_kind": arm.get("ttft_kind"),
                "ttft_ms": ttft["median"] if ttft else None,
                "ttft_min_ms": ttft["min"] if ttft else None,
                "ttft_p95_ms": ttft["p95"] if ttft else None,
                "ms_per_token": tpot["median"] if tpot else None,
                "tokens_per_s": tokens_per_s,
                "achieved_gbps": achieved_gbps(tokens_per_s, bytes_per_token),
                "roof_pct": roof_percent(tokens_per_s, bytes_per_token,
                                         bandwidth),
                "bytes_per_token": bytes_per_token,
                "peak_vram_mib": vram.get("peak_used_mib"),
                "repeats": stats["n"],
                "ids_match": arm.get("ids_match"),
            })
    return rows


def render_report_md(summary: dict) -> str:
    """Render the markdown table for docs/examples/example_qwen.md.

    Refuses a summary without a tier or a device tag: 08 section 1 says a T16
    and a T32 number never share a table, and the only way to guarantee that is
    to make the stamp mandatory here.
    """
    tier = summary.get("tier")
    device_tag = summary.get("device_tag")
    if not tier:
        raise BenchmarkError(
            "refusing to render a report without a tier: pass --tier t16|t32 "
            "(a summary without one cannot be attributed to a target, and a "
            "T16 and a T32 number must never share a table)")
    if not device_tag:
        raise BenchmarkError(
            "refusing to render a report without a device tag: pass "
            "--device-tag <exact SKU>, 08 section 1 requires every number to "
            "name its card")
    rows = build_report_rows(summary)
    ctx = summary.get("context", {})
    lines = []
    a = lines.append
    a(f"<!-- generated by scripts/benchmark_qwen.py --report-md from run "
      f"{summary.get('run_id', 'unknown')}; do not retype these numbers -->")
    a("")
    a(f"**Tier {tier.upper()} / {device_tag}** -- run "
      f"`{summary.get('run_id', 'unknown')}` (UTC "
      f"{ctx.get('started_utc', 'unknown')}), artifacts "
      f"`{summary.get('run_dir', 'unknown')}`, backend "
      f"`{ctx.get('backend', 'unknown')}`, model `{ctx.get('model', 'unknown')}`, "
      f"source_quant `{ctx.get('source_quant', 'unspecified')}` "
      f"({ctx.get('source_quant_origin', 'unspecified')}), ttft_kind "
      f"`{ctx.get('ttft_kind', 'cold')}`, max-new-tokens "
      f"{ctx.get('max_new_tokens', 'n/a')}, repeat {ctx.get('repeat', 'n/a')}.")
    a("")
    bandwidth = summary.get("bandwidth_gbps")
    # Stored KV dtype of the whole run (opt_plan/05 Q3); a bf16 cache halves the
    # KV term of the documented byte model, which is the point of the dtype arm.
    summary_kv_dtype = summary.get("kv_dtype")
    a("TTFT is the CLI's prefill wall time (weight upload and engine creation "
      "excluded: they are reported separately by the CLI). ms/token is the "
      "CLI's mean decode-step request wall; tokens/s is its reciprocal "
      "(1000 / ms/token), not a mean of per-sample throughputs. peak VRAM is a "
      "coarse nvidia-smi poll (a sampled maximum, not an instantaneous peak); "
      "it is `n/a` without --vram or without nvidia-smi.")
    a("")
    a(f"decode GB/s and roof % use the MEASURED --bandwidth "
      f"{_fmt(bandwidth, 1)} GB/s stored with this run. " + ROOF_MODEL_NOTE
      + ".")
    a("")
    a("| " + " | ".join(REPORT_COLUMNS) + " |")
    a("|" + "---|" * len(REPORT_COLUMNS))
    for row in rows:
        label = row["version"] if row["which"] == "new" \
            else f"{row['version']} (baseline)"
        a("| " + " | ".join((
            f"{label} / {row['arm']}",
            row["which"],
            row["backend"],
            f"`{row['model']}`",
            row["source_quant"],
            f"{row['tier'].upper()} / {row['device_tag']}",
            str(row["max_context"]),
            f"{row['prompt_length']} (ctx {row['context_length']})",
            _fmt(row["ttft_ms"], 1),
            _fmt(row["ms_per_token"], 3),
            _fmt(row["tokens_per_s"], 2),
            _fmt(row["achieved_gbps"], 1),
            _fmt(row["roof_pct"], 1),
            _fmt(row["peak_vram_mib"], 0),
            f"{row['repeats']} ({row['ttft_kind']})",
            _ids_match_text(row["ids_match"]),
        )) + " |")
    if not rows:
        a("")
        a("_no arms in this summary_")
    a("")
    return "\n".join(lines)


def emit_report(summary: dict, destination: str | None) -> None:
    """Write the report to stdout ("-"/None) or to `destination`."""
    text = render_report_md(summary)
    if destination in (None, "-"):
        print(text)
        return
    path = Path(destination)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Markdown report written: {path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# What "cold" and "warm" TTFT mean here. Repeated in the summary, the markdown
# and the log line so a reader can never mistake the warm bucket for an
# in-process second prefill that this CLI cannot perform.
TTFT_SEMANTICS = {
    "cold": (
        "one process per repetition with a fresh engine; weight upload and "
        "engine creation are inside TTFT, shader compilation happens before it "
        "and is reported separately by the CLI"),
    "warm": (
        "example_qwen performs exactly ONE prefill per process and has no "
        "in-process prefill repetition, so a true in-process warm prefill is "
        "not expressible with this CLI. Each measured repetition runs AFTER one "
        "discarded first process of the same arm and executable: the engine is "
        "created again (weight upload and engine creation stay inside TTFT) and "
        "only the on-disk shader/pipeline cache, the page cache and the driver "
        "caches are warm"),
}


def _resolve_path(text: str) -> Path:
    """Repo-relative CLI paths are resolved against REPO_ROOT, like --exe."""
    path = Path(text)
    return path if path.is_absolute() else REPO_ROOT / path


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
    p.add_argument("--model", default=None,
                   help="GGUF checkpoint override. Default: the version's own checkpoint "
                        "(dsl-q8 -> data/Qwen3.5-4B-Q8_0.gguf, every other version -> "
                        "data/Qwen3.5-4B-BF16.gguf). A Q8_0 checkpoint and a non-q8 "
                        "version is refused by the CLI")
    p.add_argument("--w4-pack", default=None,
                   help="offline W4 pack directory passed to every dsl-w4 run "
                        "(required by that variant; the harness never builds one: use "
                        "example_qwen --export-w4-pack)")
    p.add_argument("--w4-prefill-in-kernel", action="store_true",
                   help="pass --w4-prefill in-kernel to dsl-w4 (plan option (b) A/B "
                        "arm: W4 prefill with in-kernel dequantization). The default "
                        "is option (c), W4 decode + FP8 prefill copy")
    p.add_argument("--projection-pack", default=None,
                help="offline MERGED input-projection pack directory passed to every "
                     "non-W4 version (opt_plan/03 section 3 + D5). The CLI verifies the "
                     "pack's source identity, payload hashes, segment geometry and "
                     "format, then replaces the 120 source tensors with the 32 merged "
                     "matrices (no duplicate residency). The harness never builds one: "
                     "use example_qwen --export-projection-pack. dsl-w4 refuses a pack")
    p.add_argument("--merged-decode", dest="merged_decode", action="store_true",
                default=True,
                help="run the merged decode GEMVs when --projection-pack is given "
                     "(default)")
    p.add_argument("--no-merged-decode", dest="merged_decode", action="store_false",
                help="A/B arm: keep the per-projection decode GEMVs on the same pack")
    p.add_argument("--tokens", default="198",
                   help="comma-separated prompt token ids (used when no "
                        "--prompt-lengths/--context-lengths sweep is requested)")
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="greedy decode budget per run")
    p.add_argument("--kv-dtype", choices=("f32", "bf16", "fp8"), default=None,
                   help="KV cache stored dtype to request from the engine "
                        "(--kv-dtype; f32 = legacy position-major cache, bf16 = "
                        "two RNE codes per word, fp8 = four E4M3 codes per word "
                        "plus one FP32 scale per (kv_head, 64-position block), "
                        "both in the 64-position blocked layout). None leaves the "
                        "binary default (f32). The KV byte/capacity model follows "
                        "this dtype, so a bf16/fp8 arm "
                        "halves both the roof's KV term and the per-token "
                        "capacity cost.")
    p.add_argument("--max-context", type=int, default=64,
                   help="KV capacity in tokens (a sweep raises this to the "
                        "largest requested context plus the decode budget)")
    p.add_argument("--decode-flight", type=int, default=None,
                   help="optional example_qwen --decode-flight passthrough")
    p.add_argument("--cuda-graph-while", type=int, default=None,
                   help="optional example_qwen --cuda-graph-while passthrough: run "
                        "up to N decode tokens in ONE conditional WHILE graph launch "
                        "that stops on the device (EOS / context limit / N). Needs "
                        "the CudaGraphExt conditional-node capability AND the "
                        "engine's own instantiate/execute probe; a refusal keeps the "
                        "per-token graph and is reported in the run log")
    p.add_argument("--graph-flight", action="store_true",
                   help="the same WHILE chunk sized by --decode-flight")
    p.add_argument("--pinned-transfers", type=int, choices=(0, 1), default=None,
                   help="optional example_qwen --pinned-transfers passthrough "
                        "(1 = page-locked download staging when PinnedMemoryExt "
                        "exists, 0 = the pageable/download-pool A/B arm)")
    p.add_argument("--pdl", type=int, choices=(0, 1), default=None,
                   help="optional example_qwen --pdl passthrough: request "
                        "programmatic dependent launch at device creation. Inert "
                        "below compute capability 9.0; the engine reports the probe "
                        "and the relaxed/strict counters")
    p.add_argument("--repeat", type=int, default=argparse.SUPPRESS,
                   help=f"repetitions per (arm, executable); default "
                        f"{DEFAULT_REPEAT}, or {SWEEP_REPEAT_DEFAULT} for a "
                        f"sweep. A sweep refuses fewer than "
                        f"{SWEEP_REPEAT_DEFAULT} and any run refuses fewer than "
                        f"{MIN_REPEAT}: a single run is never reported")
    p.add_argument("--warmup", type=int, default=0,
                   help="warmup runs per (version, executable) executed "
                        "BEFORE the measured repetitions; excluded from "
                        "all statistics/pairing but fully parsed and "
                        "correctness-checked (identical generated ids)")
    p.add_argument("--profile", action="store_true",
                   help="additionally run one --benchmark-csv pass per "
                        "version on --exe (synchronized host-wall profiling); "
                        "not available together with a sweep")
    p.add_argument("--output-dir", default=".kimix_cache/qwen_benchmark",
                   help="artifact directory (raw logs preserved per run id)")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="per-process timeout in seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full sequential plan and exit; run nothing "
                        "(works with no GPU, no model file and no binary)")

    sweep = p.add_argument_group(
        "sweep", "deterministic prompt-length / context-length arms")
    sweep.add_argument("--prompt-lengths", default=None, metavar="L,L,...",
                       help="synthesize one arm per prompt length and drive "
                            "--tokens with a DETERMINISTIC token-id list of "
                            "that length (never a UTF-8 string: its token "
                            "count depends on the tokenizer). Rule: "
                            "state = (0x0C0FFEE5EEDBEEF01 ^ (length * "
                            "0xD1B54A32D192ED03)) mod 2**64, then per id "
                            "state = (state * 6364136223846793005 + "
                            "1442695040888963407) mod 2**64 and id = 1000 + "
                            "((state >> 32) mod 199000) -- identical on every "
                            "run, machine and Python version, and always "
                            "inside the Qwen3.5-4B vocabulary (248320)")
    sweep.add_argument("--context-lengths", default=None, metavar="L,L,...",
                       help="one arm per context length C: prefill C tokens so "
                            "the decode loop runs at ctx ~= C and TPOT is "
                            "attributed to that bucket. --max-context is raised "
                            "to cover the largest C plus the decode budget, and "
                            "the run aborts when the binary reports a smaller "
                            "max_context")
    sweep.add_argument("--max-argv-chars", type=int,
                       default=DEFAULT_MAX_ARGV_CHARS,
                       help="refuse an arm before launching when its projected "
                            "command line is longer than this (Windows caps a "
                            "whole command line at 32767 characters). A long "
                            "prompt is normally moved off the command line "
                            "first: --synthetic-prompt-len for a synthetic arm, "
                            "--tokens-file for an explicit --tokens list")

    tier_group = p.add_argument_group(
        "tiers and roof",
        "08-target-hardware.md section 1 says a T16 and a T32 number never "
        "share a table")
    tier_group.add_argument("--tier", choices=TIERS, default=None,
                            help="target tier the run belongs to; stamped "
                                 "into every sample/arm/summary and REQUIRED "
                                 "by --report-md. With it, prompt/context "
                                 "lengths are gated against the 08 section 5 "
                                 "ceiling for the VERSION's weight format "
                                 "(T16: 32768 in BF16, 131072 in FP8/Q8_0/W4; "
                                 "T32: 262144 in all of them)")
    tier_group.add_argument("--device-tag", default=None, metavar="SKU",
                            help="the exact card the run was measured on (for "
                                 "example 'RTX_4080_SUPER_16G'); stamped "
                                 "everywhere and REQUIRED by --report-md")
    tier_group.add_argument("--bandwidth", type=float, default=None,
                            metavar="GBPS",
                            help="MEASURED achievable DRAM bandwidth in decimal "
                                 "GB/s (tests/benchmark/benchmark_qwen_decode "
                                 "--calibrate, 08 section 7 step 7). The only "
                                 "denominator used for a roof %% column; "
                                 "without it every roof value is n/a, never a "
                                 "vendor number")

    measure = p.add_argument_group("measurement")
    measure.add_argument("--ttft-kind", choices=("cold", "warm"), default="cold",
                         help="cold = one process per repetition with a fresh "
                              "engine; warm = every measured repetition runs "
                              "after one discarded first process of the same "
                              "arm (the CLI has no in-process second prefill, so "
                              "the engine is still re-created -- see the module "
                              "docstring). Cold and warm samples never share a "
                              "bucket")
    measure.add_argument("--vram", action="store_true",
                         help="poll `nvidia-smi --query-gpu=index,memory.used,"
                              "memory.total,utilization.gpu,clocks.sm,"
                              "power.draw,temperature.gpu "
                              "--format=csv,noheader,nounits` before, during and "
                              "after each run and store the COARSE samples plus "
                              "the free-VRAM pre-flight value and the "
                              "clock/power/temperature throttle probe in the "
                              "summary; without nvidia-smi every field is null "
                              "and the run continues")
    measure.add_argument("--idle-check", dest="idle_check",
                         action="store_true", default=argparse.SUPPRESS,
                         help="refuse to launch (exit 3) when the pre-run "
                              "nvidia-smi sample shows another tenant on the GPU "
                              "(utilization.gpu above --gpu-idle-util-pct); 08 "
                              "section 9 H3. This is the default")
    measure.add_argument("--no-idle-check", dest="idle_check",
                         action="store_false", default=argparse.SUPPRESS,
                         help="record the foreign-activity warning instead of "
                              "refusing to launch (for desktops whose compositor "
                              "keeps the GPU slightly busy)")
    measure.add_argument("--gpu-idle-util-pct", type=float,
                         default=DEFAULT_GPU_IDLE_UTIL_PCT,
                         help="pre-run utilization.gpu above this counts as "
                              "foreign GPU activity")
    measure.add_argument("--gpu-idle-max-used-mib", type=int,
                         default=DEFAULT_GPU_IDLE_MAX_USED_MIB,
                         help="also refuse to launch when memory.used exceeds "
                              "this many MiB; 0 disables the check (the desktop "
                              "always holds some VRAM)")
    measure.add_argument("--min-free-vram-mib", type=int,
                         default=DEFAULT_MIN_FREE_VRAM_MIB,
                         help="abort BEFORE launching anything when nvidia-smi "
                              "reports less free VRAM than this")
    measure.add_argument("--vram-interval", type=float,
                         default=DEFAULT_VRAM_INTERVAL_S,
                         help="seconds between VRAM samples (coarse by "
                              "construction: each sample is an nvidia-smi "
                              "process launch)")
    measure.add_argument("--source-quant", default=None,
                         help="SOURCE quantization of the checkpoint (bf16, "
                              "q8_0, ...). Mandatory field in every summary and "
                              "report row so a double-quantized Q8_0-sourced arm "
                              "can never be silently mixed with a BF16-sourced "
                              "arm. Default: inferred from an explicit BF16/Q8_0 "
                              "file-name marker, and the inference itself is "
                              "recorded")

    ident = p.add_argument_group("identity and report")
    ident.add_argument("--require-id-match", default=None,
                       metavar="REFERENCE.json",
                       help="hard-fail any arm whose generated ids differ from "
                            "this reference file (naming the arm and the first "
                            "differing token index); an arm the reference does "
                            "not cover also fails. Schema: see the module "
                            "docstring")
    ident.add_argument("--write-id-reference", default=None,
                       metavar="REFERENCE.json",
                       help="write the reference file from this run (after "
                            "every check passed), so a later --require-id-match "
                            "proves an optimization did not change the model")
    ident.add_argument("--report-md", nargs="?", const="-", default=None,
                       metavar="PATH",
                       help="also emit the docs/examples/example_qwen.md "
                            "markdown table from the collected JSON; without a "
                            "PATH (or with '-') it goes to stdout")
    ident.add_argument("--report-from", default=None, metavar="SUMMARY.json",
                       help="run nothing: render the markdown report from an "
                            "existing summary.json")
    return p.parse_args(argv)


def validate_args(args) -> dict:
    """Validate and resolve everything derivable from the CLI, fail closed.

    Returns the resolved run configuration (versions, arms, max_context,
    repeat, source_quant) and writes the resolved repeat back onto `args` so
    the dry run and the summary show one identical number.
    """
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not versions:
        raise BenchmarkError("--versions is empty")
    for v in versions:
        if v not in KNOWN_VERSIONS:
            raise BenchmarkError(
                f"unknown version {v!r}; supported: {', '.join(KNOWN_VERSIONS)}")
    if len(set(versions)) != len(versions):
        raise BenchmarkError("--versions contains duplicates")

    # Checkpoint/version consistency (phase 8a): dsl-q8 only exists for a Q8_0
    # checkpoint and every other version only for a BF16/F32 one. The CLI fails
    # closed on a mismatch, so the harness refuses the arm before launching
    # anything instead of writing a refusal log per repetition.
    model_is_q8 = bool(args.model) and "q8_0" in args.model.lower()
    mixed_q8 = any(v == "dsl-q8" for v in versions) and any(v != "dsl-q8" for v in versions)
    if args.model and mixed_q8:
        raise BenchmarkError(
            "--model is a single checkpoint, but --versions mixes dsl-q8 (a Q8_0 "
            "checkpoint) with versions that need the BF16 checkpoint; run them as "
            "separate --versions groups, or omit --model so each version loads its "
            "own checkpoint")
    if args.model and model_is_q8 and not all(v == "dsl-q8" for v in versions):
        raise BenchmarkError(
            f"--model {args.model!r} looks like a Q8_0 checkpoint, but the requested "
            "versions need the BF16 checkpoint; use --versions dsl-q8, or omit --model")

    if args.max_new_tokens < 1:
        raise BenchmarkError("--max-new-tokens must be >= 1")
    if args.warmup < 0:
        raise BenchmarkError("--warmup must be >= 0")
    if args.decode_flight is not None and args.decode_flight < 0:
        raise BenchmarkError("--decode-flight must be >= 0")
    if args.cuda_graph_while is not None and args.cuda_graph_while < 0:
        raise BenchmarkError("--cuda-graph-while must be >= 0 (0 disables it)")
    if args.cuda_graph_while is not None and args.cuda_graph_while > 0 \
            and args.decode_flight == 0:
        raise BenchmarkError("--cuda-graph-while needs in-flight decoding "
                             "(--decode-flight 0 disables it)")
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    if args.max_context < 1:
        raise BenchmarkError("--max-context must be >= 1")
    if args.min_free_vram_mib < 0:
        raise BenchmarkError("--min-free-vram-mib must be >= 0")
    if args.vram_interval <= 0:
        raise BenchmarkError("--vram-interval must be positive")
    if args.max_argv_chars < 256:
        raise BenchmarkError(
            "--max-argv-chars must be >= 256 (a real command line is longer)")
    # --idle-check / --no-idle-check share one dest; the default is to enforce.
    args.idle_check = getattr(args, "idle_check", True)
    tier = resolve_tier(args.tier)
    device_tag = resolve_device_tag(args.device_tag)
    bandwidth = resolve_bandwidth(args.bandwidth)
    # --report-from takes the tier from the stored summary instead (and refuses
    # a stored summary without one); a live run must name both up front.
    if args.report_md is not None and not args.report_from \
            and not (tier and device_tag):
        raise BenchmarkError(
            "--report-md requires --tier t16|t32 and --device-tag <sku>: 08 "
            "section 1 says a T16 and a T32 number never share a table, so a "
            "report that does not name both is refused")
    arms, max_context, sweeping = plan_sweep(args)
    check_tier_limits(tier, arms, max_new_tokens=args.max_new_tokens,
                      versions=versions, max_context=max_context,
                      kv_dtype=args.kv_dtype)
    if args.profile and sweeping:
        raise BenchmarkError(
            "--profile runs one profiling pass per version, so it cannot be "
            "combined with --prompt-lengths/--context-lengths; run the sweep "
            "and the profile pass separately")
    repeat = resolve_repeat(args, sweeping)
    args.repeat = repeat
    source_quant, source_quant_origin = resolve_source_quant(
        args.source_quant, resolve_arm_model(versions[0], args.model))
    return {
        "versions": versions,
        "arms": arms,
        "max_context": max_context,
        "max_context_requested": args.max_context,
    "kv_dtype": getattr(args, "kv_dtype", None),
        "sweeping": sweeping,
        "repeat": repeat,
        "source_quant": source_quant,
        "source_quant_origin": source_quant_origin,
        "tier": tier,
        "device_tag": device_tag,
        "bandwidth_gbps": bandwidth,
        "tier_usable_gib": TIER_USABLE_GIB.get(tier) if tier else None,
        # The tier's ceiling across formats, plus the ceiling the gate actually
        # applied to each requested version (08 section 5 is per format).
        "tier_max_context": TIER_MAX_CONTEXT.get(tier) if tier else None,
        "tier_version_max_context": (
            {v: tier_format_max_context(tier, weight_format_for_version(v),
                                  args.kv_dtype)
             for v in versions} if tier else None),
  }


def _tokens_file_for(run_dir: Path, version: str, arm: dict) -> Path:
    """Where an over-long explicit id list is written (one file per arm/version
    so a baseline binary sees the identical file)."""
    return run_dir / f"tokens_{arm['key']}_{version}.txt"


def _plan_entry(args, config, *, version: str, arm: dict, which: str,
                kind: str, repetition: int | None, exe_path: Path,
                baseline_path: Path | None, csv_path: Path | None,
                run_dir: Path) -> dict:
    """One process of the strictly sequential plan."""
    exe = str(baseline_path if which == "baseline" else exe_path)
    suffix = {
        "normal": f"rep{repetition}",
        "warmup": f"warmup{repetition}",
        "cache-warm": "cachewarm",
        "profile": "profile",
    }[kind]
    label = f"{version}_{which}_{arm['key']}_{suffix}"
    base = build_command(exe, version, args, csv_path, prompt=("none", None),
                         max_context=config["max_context"])
    prompt_kind, prompt_value = prompt_transport(
        arm, base_cmd=base, max_chars=args.max_argv_chars)
    tokens_file = None
    if prompt_kind == "file":
        path = _tokens_file_for(run_dir, version, arm)
        prompt_value = str(path)
        tokens_file = {"path": str(path),
                       "text": "".join(f"{i}\n" for i in
                                       arm["tokens"].split(","))}
    command = build_command(exe, version, args, csv_path,
                            prompt=(prompt_kind, prompt_value),
                            max_context=config["max_context"])
    return {
        "label": label,
        "exe": exe,
        "which": which,
        "version": version,
        "arm": arm,
        "kind": kind,
        "repetition": repetition,
        "csv_path": csv_path,
        "max_context": config["max_context"],
        "prompt_transport": prompt_kind,
        "tokens_file": tokens_file,
        "command": command,
  }


def build_plan(args, config, *, exe_path: Path, baseline_path: Path | None,
               run_dir: Path) -> list[dict]:
    """Every process of this run, in execution order (never concurrent).

    Per arm: the user's warmups (excluded from statistics), then -- for
    --ttft-kind warm without any --warmup -- one discarded warming process per
    executable, then the measured repetitions with the baseline interleaved
    (baseline,new then new,baseline).
    """
    plan: list[dict] = []
    for arm in config["arms"]:
        for version in config["versions"]:
            base_order = ["baseline", "new"] if baseline_path else ["new"]
            for w in range(args.warmup):
                for which in base_order:
                    plan.append(_plan_entry(
                        args, config, version=version, arm=arm, which=which,
                        kind="warmup", repetition=w,
                        exe_path=exe_path, baseline_path=baseline_path,
                        csv_path=None, run_dir=run_dir))
            if args.ttft_kind == "warm" and args.warmup == 0:
                for which in base_order:
                    plan.append(_plan_entry(
                        args, config, version=version, arm=arm, which=which,
                        kind="cache-warm", repetition=None,
                        exe_path=exe_path, baseline_path=baseline_path,
                        csv_path=None, run_dir=run_dir))
            for rep in range(config["repeat"]):
                order = list(base_order)
                if baseline_path and rep % 2 == 1:
                    order = ["new", "baseline"]
                for which in order:
                    plan.append(_plan_entry(
                        args, config, version=version, arm=arm, which=which,
                        kind="normal", repetition=rep,
                        exe_path=exe_path, baseline_path=baseline_path,
                        csv_path=None, run_dir=run_dir))
        for version in config["versions"]:
            if not args.profile:
                continue
            csv_path = run_dir / f"profile_{version}_{arm['key']}.csv"
            plan.append(_plan_entry(
                args, config, version=version, arm=arm, which="new",
                kind="profile", repetition=None, exe_path=exe_path,
                baseline_path=baseline_path, csv_path=csv_path,
                run_dir=run_dir))
    return plan


def merge_arm_vram(records: list[dict]) -> dict | None:
    """One coarse VRAM record per arm: the worst sampled peak of its
    repetitions (per-run samples stay in summary["samples"])."""
    records = [r for r in records if r]
    if not records:
        return None
    peaks = [r["peak_used_mib"] for r in records
             if r.get("peak_used_mib") is not None]
    loads = [r["after_load_prefill_mib"] for r in records
             if r.get("after_load_prefill_mib") is not None]
    merged = dict(records[0])
    merged["peak_used_mib"] = max(peaks) if peaks else None
    merged["after_load_prefill_mib"] = max(loads) if loads else None
    merged["after_decode_mib"] = records[-1].get("after_decode_mib")
    merged["after_exit_mib"] = records[-1].get("after_exit_mib")
    merged["before_start_mib"] = records[0].get("before_start_mib")
    merged["sample_count"] = sum(r.get("sample_count", 0) for r in records)
    merged["per_run_peak_mib"] = [r.get("peak_used_mib") for r in records]
    merged.pop("samples", None)
    return merged


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = validate_args(args)
    except BenchmarkError as e:
        print(f"benchmark_qwen: error: {e}", file=sys.stderr)
        return 1

    # ---- offline report ----------------------------------------------------
    if args.report_from:
        path = Path(args.report_from)
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            summary_in = json.loads(path.read_text(encoding="utf-8"))
        except OSError as e:
            print(f"benchmark_qwen: error: --report-from cannot read {path}: {e}",
                  file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"benchmark_qwen: error: --report-from {path} is not JSON: {e}",
                  file=sys.stderr)
            return 1
        if not isinstance(summary_in, dict) or "arms" not in summary_in:
            print(f"benchmark_qwen: error: --report-from {path} is not a "
                  f"benchmark_qwen summary (no 'arms')", file=sys.stderr)
            return 1
        # A stored summary may be re-rendered only as the tier it was measured
        # on: --tier on this command line must either be absent or identical.
        stored_tier = summary_in.get("tier")
        if args.tier and stored_tier and resolve_tier(args.tier) != stored_tier:
            print(f"benchmark_qwen: error: --report-from {path} was measured on "
                  f"tier {stored_tier}, but --tier {args.tier} was requested; a "
                  f"T16 and a T32 table are never mixed", file=sys.stderr)
            return 1
        if args.tier and not stored_tier:
            summary_in = {**summary_in, "tier": resolve_tier(args.tier)}
        if args.device_tag and not summary_in.get("device_tag"):
            summary_in = {**summary_in,
                          "device_tag": resolve_device_tag(args.device_tag)}
        if args.bandwidth is not None:
            summary_in = {**summary_in,
                          "bandwidth_gbps": resolve_bandwidth(args.bandwidth)}
        try:
            emit_report(summary_in, args.report_md)
        except BenchmarkError as e:
            print(f"benchmark_qwen: error: {e}", file=sys.stderr)
            return 1
        return 0

    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = REPO_ROOT / exe_path
    baseline_path = None
    if args.baseline_exe:
        baseline_path = Path(args.baseline_exe)
        if not baseline_path.is_absolute():
            baseline_path = REPO_ROOT / baseline_path
    model_path = Path(resolve_arm_model(config["versions"][0], args.model))
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    run_dir = out_dir / run_id

    plan = build_plan(args, config, exe_path=exe_path,
                      baseline_path=baseline_path, run_dir=run_dir)

    # ---- dry run (no GPU, no model, no binary required) --------------------
    if args.dry_run:
        print("DRY RUN -- nothing will be executed. Sequential plan:")
        print(f"  cwd: {REPO_ROOT}")
        print(f"  output dir: {out_dir} (run dir {run_dir})")
        print(f"  backend: {args.backend} | versions: "
              f"{', '.join(config['versions'])}")
        print(f"  arms: {', '.join(a['key'] for a in config['arms'])}")
        print(f"  max-context to pass: {config['max_context']} "
              f"(--max-context {config['max_context_requested']}"
              + (", raised to cover the largest requested context plus the "
                 "decode budget)" if config["max_context"] !=
                 config["max_context_requested"] else ")"))
        print(f"  repeat: {config['repeat']} | warmup: {args.warmup} | "
              f"ttft-kind: {args.ttft_kind} | source_quant: "
              f"{config['source_quant']} ({config['source_quant_origin']})")
        print(f"  tier: {config['tier'] or 'NOT DECLARED (a report will refuse it)'}"
              f" | device tag: {config['device_tag'] or 'NOT DECLARED'}"
              f" | measured bandwidth: "
              f"{config['bandwidth_gbps'] if config['bandwidth_gbps'] else 'not given (roof % = n/a)'}"
              f" GB/s")
        if config["tier"]:
            ceilings = config.get("tier_version_max_context") or {}
            print(f"  tier limits: usable {config['tier_usable_gib']} GiB | "
                  f"context ceiling per version (08 section 5, per weight "
                  f"format): "
                  + ", ".join(f"{v} <= {cap}" for v, cap in ceilings.items())
                  + f" tokens (tier maximum across formats "
                  f"{config['tier_max_context']})")
        if args.ttft_kind == "warm":
            print(f"  ttft-kind warm semantics: {TTFT_SEMANTICS['warm']}")
        print(f"  vram sampling: "
              + (f"on, every {args.vram_interval}s, abort below "
                 f"{args.min_free_vram_mib} MiB free"
                 if args.vram else "off (pre-flight free-VRAM check only)"))
        print(f"  idle pre-flight: "
              + (f"enforced at utilization.gpu > {args.gpu_idle_util_pct}%"
                 + (f", memory.used > {args.gpu_idle_max_used_mib} MiB"
                    if args.gpu_idle_max_used_mib > 0 else "")
                 if args.idle_check else
                 "recording only (--no-idle-check)"))
        if args.require_id_match:
            print(f"  id reference required: {args.require_id_match}")
        if args.write_id_reference:
            print(f"  id reference to write: {args.write_id_reference}")
        if args.report_md is not None:
            print(f"  markdown report: "
                  f"{args.report_md if args.report_md != '-' else 'stdout'}")
        for warning, path in (("exe", exe_path), ("model", model_path)):
            if not path.exists():
                print(f"  WARNING: {warning} not found: {path}")
        if baseline_path and not baseline_path.exists():
            print(f"  WARNING: baseline exe not found: {baseline_path}")
        for arm in config["arms"]:
            ids = arm["tokens"].split(",")
            transports = sorted({entry["prompt_transport"]
                                 for entry in plan if entry["arm"] is arm})
            print(f"  prompt {arm['key']} ({arm['origin']}): {len(ids)} token "
                  f"ids, transport {'/'.join(transports) or 'tokens'}, first 8 = "
                  f"{', '.join(ids[:8])}"
                  + ("" if len(ids) <= 8 else ", ..."))
        for i, entry in enumerate(plan, 1):
            print(f"  {i:2d}. [{entry['label']}] {' '.join(entry['command'])}")
            problem = command_length_error(entry["command"],
                                           max_chars=args.max_argv_chars,
                                           label=entry["label"])
            if problem:
                print(f"      WARNING: {problem}")
            if entry["tokens_file"]:
                print(f"      note: the ids are written to "
                      f"{entry['tokens_file']['path']} immediately before launch "
                      f"(nothing is written by --dry-run)")
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

    # Fail before burning GPU time on a reference that cannot be satisfied.
    reference = None
    if args.require_id_match:
        try:
            reference = load_id_reference(_resolve_path(args.require_id_match))
        except BenchmarkError as e:
            print(f"benchmark_qwen: error: {e}", file=sys.stderr)
            return 1

    # ---- host provenance + pre-flight -------------------------------------
    gpu_index = 0
    gpu_identity = query_gpu_identity(gpu_index=gpu_index)
    if gpu_identity.get("nvidia_smi_available"):
        print(f"[gpu] {gpu_identity['name']} | cc "
              f"{gpu_identity['compute_capability']} | driver "
              f"{gpu_identity['driver_version']} | tier "
              f"{config['tier'] or 'undeclared'} / device tag "
              f"{config['device_tag'] or 'undeclared'}")
    else:
        print("[gpu] nvidia-smi unavailable: GPU identity, VRAM and the idle "
              "pre-flight are recorded as null and the run continues")
    preflight_state = query_gpu_state(gpu_index=gpu_index)
    preflight_memory = query_gpu_memory(gpu_index=gpu_index)
    preflight = preflight_vram(
        (preflight_memory or {}).get("free_mib"),
        min_free_mib=args.min_free_vram_mib, gpu=gpu_identity)
    print(f"[vram] {preflight['message']}")
    # The idle decision needs the full sample; when only the memory query is
    # available (older callers, tests) fall back to it so the free-VRAM refusal
    # is never lost.
    idle_state = preflight_state
    if idle_state is None and preflight_memory:
        idle_state = {"gpu_index": gpu_index,
                      "used_mib": preflight_memory.get("used_mib"),
                      "total_mib": preflight_memory.get("total_mib"),
                      "utilization_pct": None, "sm_clock_mhz": None,
                      "power_w": None, "temperature_c": None}
    idle = preflight_gpu_idle(
        idle_state, min_free_mib=args.min_free_vram_mib,
        max_util_pct=args.gpu_idle_util_pct,
        max_used_mib=args.gpu_idle_max_used_mib,
        enforced=args.idle_check, gpu=gpu_identity)
    print(f"[idle] {idle['message']}")
    # Two independent refusals (08 section 9 H3): free VRAM below the floor and
    # foreign GPU activity. The idle message carries the detail when it fired.
    if preflight["status"] == "low" or idle["status"] in ("low", "busy"):
        message = idle["message"] if idle["status"] != "ok" else preflight["message"]
        print(f"benchmark_qwen: ABORTING: {message}", file=sys.stderr)
        return 3

    if args.ttft_kind == "warm":
        print(f"[ttft] warm: {TTFT_SEMANTICS['warm']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()

    samples: list[dict] = []
    warmups: list[dict] = []
    profiles: dict = {}
    all_commands: list[dict] = []
    gpu_during_samples: list[dict] = []

    try:
        for entry in plan:
            cmd = entry["command"]
            label = entry["label"]
            problem = command_length_error(cmd, max_chars=args.max_argv_chars,
                                           label=label)
            if problem:
                raise BenchmarkError(problem)
            # An over-long explicit id list travels as a file that is written
            # here -- immediately before the process that reads it -- so
            # --dry-run never creates artifacts.
            if entry["tokens_file"]:
                path = Path(entry["tokens_file"]["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(entry["tokens_file"]["text"], encoding="utf-8")
            before_state = query_gpu_state(gpu_index=gpu_index) \
                if args.vram else None
            before_mib = (before_state or {}).get("used_mib") \
                if args.vram else None
            poller = VramPoller(args.vram_interval, gpu_index=gpu_index) \
                if args.vram else None
            if poller is not None:
                poller.start()
            try:
                result = run_process(cmd, run_dir / label, timeout=args.timeout,
                                     label=label)
            finally:
                during = poller.stop() if poller is not None else []
            gpu_during_samples.extend(during)
            after_state = query_gpu_state(gpu_index=gpu_index) \
                if args.vram else None
            after_mib = (after_state or {}).get("used_mib") \
                if args.vram else None
            all_commands.append({"label": label, "command": cmd})
            arm = entry["arm"]
            metrics = parse_run_output(
                result["stdout"], version=entry["version"],
                prompt_len=arm["prompt_length"],
                max_new_tokens=args.max_new_tokens, label=label)
            engine = parse_engine_lines(result["stdout"], label)
            sample = {
                "label": label,
                "version": entry["version"],
                "which": entry["which"],
                "arm": arm["key"],
                "prompt_length": arm["prompt_length"],
                "context_length": arm["context_length"],
                "max_context": entry["max_context"],
                "ttft_kind": args.ttft_kind,
                "source_quant": config["source_quant"],
                "tier": config["tier"],
                "device_tag": config["device_tag"],
                "bandwidth_gbps": config["bandwidth_gbps"],
                "prompt_transport": entry["prompt_transport"],
                "kind": entry["kind"],
                "repetition": entry["repetition"],
                "command": cmd,
                "cwd": result["cwd"],
                "returncode": result["returncode"],
                "process_wall_s": result["process_wall_s"],
                "stdout_log": result["stdout_log"],
                "stderr_log": result["stderr_log"],
                "metrics": metrics,
                "engine": engine,
                "vram": (summarize_vram_samples(
                    before_start_mib=before_mib, during_samples=during,
                    after_exit_mib=after_mib,
                    interval_s=args.vram_interval,
                    before_state=before_state, after_state=after_state)
                    if args.vram else None),
                "throttle": (summarize_throttle(
                    before_state=before_state, after_state=after_state,
                    during_samples=during) if args.vram else None),
            }
            if entry["kind"] == "normal":
                samples.append(sample)
            elif entry["kind"] in ("warmup", "cache-warm"):
                warmups.append(sample)
            else:
                bench = parse_benchmark_stdout(result["stdout"], label)
                rows = parse_profile_csv(entry["csv_path"],
                                         version=entry["version"],
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
                profiles[entry["version"]] = {
                    "arm": arm["key"],
                    "sample": sample,
                    "csv": str(entry["csv_path"]),
                    "stdout_benchmark": bench,
                    "decode_aggregates": agg,
                }

        # ---- statistics / determinism / pairing per arm --------------------
        kinds = {s["ttft_kind"] for s in samples}
        if len(kinds) > 1:
            raise BenchmarkError(
                f"cold and warm TTFT samples must never share a bucket, got "
                f"{sorted(kinds)}")
        grouped = group_samples_by_arm(samples)
        warm_grouped = group_samples_by_arm(warmups)
        arm_records: list[dict] = []
        for arm in config["arms"]:
            for version in config["versions"]:
                key = (version, arm["prompt_length"], arm["context_length"])
                buckets = grouped.get(
                    key, {"new": [], "baseline": []})
                wbuckets = warm_grouped.get(
                    key, {"new": [], "baseline": []})
                if not buckets["new"]:
                    raise BenchmarkError(
                        f"arm {arm['key']}/{version}: no measured samples")
                record: dict = {
                    "arm": arm["key"],
                    "version": version,
                    "prompt_length": arm["prompt_length"],
                    "context_length": arm["context_length"],
                    "max_context": config["max_context"],
                    "ttft_kind": args.ttft_kind,
                    "source_quant": config["source_quant"],
                    "source_quant_origin": config["source_quant_origin"],
                    "backend": args.backend,
                    "model": str(model_path),
                    "repeat": config["repeat"],
                    # Tier/device/bandwidth are stamped on the arm too: a row
                    # can then never be copied out of its tier's table.
                    "tier": config["tier"],
                    "device_tag": config["device_tag"],
                    "bandwidth_gbps": config["bandwidth_gbps"],
                    "stats": {},
                }
                ref_ids = None
                for which in ("new", "baseline"):
                    ws = wbuckets[which]
                    if not ws:
                        continue
                    wids = check_determinism(
                        ws, f"{version}/{arm['key']}/{which}/warmup")
                    if ref_ids is not None and wids != ref_ids:
                        raise BenchmarkError(
                            f"{version}/{arm['key']}: warmup and measured runs "
                            f"generate different ids ({wids} vs {ref_ids}); "
                            f"non-deterministic generation, run invalid")
                    ref_ids = wids
                for which in ("new", "baseline"):
                    ss = buckets[which]
                    if not ss:
                        continue
                    ids = check_determinism(ss, f"{version}/{arm['key']}/{which}")
                    if ref_ids is not None and ids != ref_ids:
                        raise BenchmarkError(
                            f"{version}/{arm['key']}: baseline and new "
                            f"executables generate different ids "
                            f"({ref_ids} vs {ids}); comparison invalid")
                    ref_ids = ids
                    record["stats"][which] = summarize_samples(ss)
                    record.setdefault("sample_labels", {})[which] = [
                        s["label"] for s in ss]
                first = buckets["new"][0]
                record["generated_ids"] = ref_ids
                record["stop"] = first["metrics"]["stop"]
                record["engine"] = first["engine"]
                record["warmup_checked"] = {
                    which: len(wbuckets[which])
                    for which in ("new", "baseline") if wbuckets[which]
                }
                # TPOT attribution: which context bucket this arm belongs to,
                # what the decode loop actually ran at, and the CLI's ms/token
                # as the bucket's TPOT.
                record["context_attribution"] = {
                    "context_bucket_tokens": arm["context_length"],
                    "prompt_length": arm["prompt_length"],
                    "observed_decode_context_tokens": (
                        first["metrics"]["prefill_tokens"]
                        + first["metrics"]["decode_steps"]),
                    "tpot_ms_per_token": record["stats"]["new"][
                        "decode_ms_per_token"],
                    "tpot_tokens_per_s": record["stats"]["new"][
                        "decode_tokens_per_s"],
                    "note": ("prefill is the prompt length, so decode step i "
                             "runs at ctx = prompt_length + i; the bucket is "
                             "the --context-lengths value and its TPOT is the "
                             "CLI's mean decode ms/token for that arm"),
                }
                # Roof of the decode step against the MEASURED bandwidth. The
                # byte model is documented (see ROOF_MODEL_NOTE) and carries the
                # CURRENT KV read amplification of 03 section 4.1; without
                # --bandwidth or for an unknown source_quant it is null.
                observed_ctx = int(first["metrics"]["prefill_tokens"] +
                                   first["metrics"]["decode_steps"])
                storage = first["engine"]["weight_storage"]
                # Which KV read amplification the MEASURED step had: the engine's own
                # `decode attn:` line decides, and only at/above its threshold. Every
                # other case keeps the legacy 4x term (03 section 4.1).
                kv_engine = first["engine"].get("decode_attn")
                kv_dedup = decode_attn_dedup_applies(first["engine"], observed_ctx)
                amplification = decode_kv_read_amplification(kv_dedup=kv_dedup)
                bytes_per_token = roof_bytes_per_token(storage, observed_ctx,
                                             kv_dedup=kv_dedup,
                                             kv_dtype=config.get("kv_dtype"))
                tps_median = record["stats"]["new"][
                    "decode_tokens_per_s"]["median"]
                record["roof"] = {
                    "weight_storage": storage,
                    "observed_context_tokens": observed_ctx,
                    "bytes_per_token": bytes_per_token,
                    "kv_dtype": config.get("kv_dtype") or "f32",
      "kv_bytes_per_token": kv_bytes_read_per_token(
          observed_ctx, kv_dedup=kv_dedup, kv_dtype=config.get("kv_dtype")),
                    "kv_read_amplification": amplification,
                    "kv_dedup": kv_dedup,
                    "kv_dedup_reported_by_engine": (
                        None if kv_engine is None
                        else kv_engine.get("active") == "split"),
                    "kv_split_threshold": (
                        None if kv_engine is None else kv_engine.get("threshold")),
                    "decode_attn": kv_engine,
                    "achieved_gbps": achieved_gbps(tps_median,
                                                   bytes_per_token),
                    "bandwidth_gbps": config["bandwidth_gbps"],
                    "roof_pct": roof_percent(tps_median, bytes_per_token,
                                             config["bandwidth_gbps"]),
                    "model": ROOF_MODEL_NOTE,
                }
                reported = first["engine"]["reported_max_context"]
                needed = arm["prompt_length"] + args.max_new_tokens
                if reported < arm["context_length"] or reported < needed:
                    raise BenchmarkError(
                        f"arm {arm['key']}/{version}: the binary reports "
                        f"max_context={reported}, which is smaller than the "
                        f"requested context length {arm['context_length']} "
                        f"(prompt {arm['prompt_length']} + "
                        f"--max-new-tokens {args.max_new_tokens} = {needed}). "
                        f"Refusing to report a TPOT measured at a context the "
                        f"engine cannot hold")
                if args.vram:
                    record["vram"] = merge_arm_vram(
                        [s["vram"] for s in buckets["new"]])
                if baseline_path:
                    record["baseline_comparison"] = paired_deltas(
                        buckets["new"], buckets["baseline"],
                        f"{version}/{arm['key']}")
                if reference is not None:
                    match = compare_ids_to_reference(
                        reference, arm=arm["key"], ids=ref_ids)
                    record["ids_match"] = match["match"]
                arm_records.append(record)

        # Final throttle sample: the after-run half of 08 section 7 step 8.
        final_state = query_gpu_state(gpu_index=gpu_index)
        if final_state:
            print(f"[gpu] after the run: sm clock "
                  f"{final_state['sm_clock_mhz']} MHz, power "
                  f"{final_state['power_w']} W, temperature "
                  f"{final_state['temperature_c']} C")

        if reference is not None:
            print(f"[ids] every arm matches the reference "
                  f"{args.require_id_match}")
        arms_by_version: dict = {}
        for record in arm_records:
            entry = arms_by_version.setdefault(
                record["version"], {"arms": [], "generated_ids": {}})
            entry["arms"].append(record["arm"])
            entry["generated_ids"][record["arm"]] = record["generated_ids"]
            entry.setdefault("stops", {})[record["arm"]] = record["stop"]
            entry["ttft_kind"] = record["ttft_kind"]

        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            # Tier / device / roof denominator: stamped into the summary, every
            # arm record and every sample so a table can never mix tiers.
            "tier": config["tier"],
            "device_tag": config["device_tag"],
            "bandwidth_gbps": config["bandwidth_gbps"],
            "tier_usable_gib": config["tier_usable_gib"],
            "tier_max_context": config["tier_max_context"],
            "tier_version_max_context": config.get("tier_version_max_context"),
            "roof_model": ROOF_MODEL_NOTE,
            # The BASELINE model constant (legacy per-q_head 4x) and, separately, what the
            # engine of this run reported about its decode attention. Per-arm effective
            # values live in each arm roof block, so a summary can never present the 4x
            # baseline as what actually ran.
            "roof_kv_read_amplification": DECODE_KV_READ_AMPLIFICATION,
            "roof_kv_read_amplification_dedup": DECODE_KV_READ_AMPLIFICATION_DEDUP,
            "roof_kv_dedup_reported_by_engine": any(
                (arm.get("roof") or {}).get("kv_dedup_reported_by_engine")
                for arm in arm_records),
            "decode_attn": {
                # Engine provenance per arm: requested/active realization, threshold,
                # grid bound and whether the measured context actually used the dedup.
                arm["arm"]: {
                    "version": arm["version"],
                    "engine": (arm.get("engine") or {}).get("decode_attn"),
                    "kv_dedup": (arm.get("roof") or {}).get("kv_dedup"),
                    "kv_read_amplification": (arm.get("roof") or {}).get(
                        "kv_read_amplification"),
                    "observed_context_tokens": (arm.get("roof") or {}).get(
                        "observed_context_tokens"),
                }
                for arm in arm_records
            },
            "context": {
                "started_utc": started_utc,
                "cwd": str(REPO_ROOT),
                "exe": str(exe_path),
                "baseline_exe": str(baseline_path) if baseline_path else None,
                "backend": args.backend,
                "model": str(model_path),
                "model_arg": args.model,
                "model_per_version": {v: resolve_arm_model(v, args.model)
                                      for v in config["versions"]},
                "tokens": args.tokens,
                "arm_labels": [a["key"] for a in config["arms"]],
                "prompt_lengths": [a["prompt_length"] for a in config["arms"]],
                "prompt_id_rule": (
                    "synthetic_prompt_ids(): 64-bit LCG seeded with "
                    "(0x0C0FFEE5EEDBEEF01 ^ length*0xD1B54A32D192ED03), "
                    "state=(state*6364136223846793005+1442695040888963407) mod "
                    "2**64, id=1000+((state>>32) mod 199000)"),
                "max_new_tokens": args.max_new_tokens,
                "max_context": config["max_context"],
                "max_context_requested": config["max_context_requested"],
                "decode_flight": args.decode_flight,
        "cuda_graph_while": args.cuda_graph_while,
        "graph_flight": bool(args.graph_flight),
        "pinned_transfers": args.pinned_transfers,
        "pdl": args.pdl,
                "repeat": config["repeat"],
                "warmup": args.warmup,
                "profile": args.profile,
                "sweeping": config["sweeping"],
                "timeout_s": args.timeout,
                "ttft_kind": args.ttft_kind,
                "ttft_semantics": TTFT_SEMANTICS[args.ttft_kind],
                "source_quant": config["source_quant"],
                "source_quant_origin": config["source_quant_origin"],
                "vram_sampling": args.vram,
                "min_free_vram_mib": args.min_free_vram_mib,
                "vram_interval_s": args.vram_interval,
                "gpu_sampling_fields": ["index", *GPU_SAMPLE_FIELDS],
                "idle_check_enforced": args.idle_check,
                "gpu_idle_util_pct": args.gpu_idle_util_pct,
                "gpu_idle_max_used_mib": args.gpu_idle_max_used_mib,
                "prompt_transports": sorted(
                    {entry["prompt_transport"] for entry in plan}),
                "id_reference_required": str(args.require_id_match)
                if args.require_id_match else None,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "execution": "strictly sequential, one process at a time",
            },
            "provenance": {
                "gpu": gpu_identity,
                "vram_preflight": preflight,
                "idle_preflight": idle,
                "gpu_state_before_start": gpu_state_flat(preflight_state),
                "throttle_probe": summarize_throttle(
                    before_state=preflight_state, after_state=final_state,
                    during_samples=gpu_during_samples),
                "vram_memory_before_start": preflight_memory,
                "gpu_index": gpu_index,
                "gpu_index_source": ("nvidia-smi ordering; CUDA_VISIBLE_DEVICES "
                                     "is not resolved by this harness"),
                "exe": file_fingerprint(exe_path),
                "baseline_exe": (file_fingerprint(baseline_path)
                                 if baseline_path else None),
                "model": file_fingerprint(model_path),
                "id_reference": (file_fingerprint(
                    _resolve_path(args.require_id_match))
                    if args.require_id_match else None),
            },
            "all_commands": all_commands,
            "arms": arm_records,
            "versions": arms_by_version,
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

        if args.write_id_reference:
            ref_doc = build_id_reference(
                backend=args.backend, model=str(model_path),
                source_quant=config["source_quant"],
                max_new_tokens=args.max_new_tokens, arms=arm_records)
            ref_path = _resolve_path(args.write_id_reference)
            write_id_reference(ref_path, ref_doc)
            print(f"ID reference written: {ref_path}")

        print(f"\nSummary written: {summary_json}")
        print(f"                 {summary_md}")
        print(f"Raw artifacts:   {run_dir}")
        if args.report_md is not None:
            try:
                emit_report(summary, args.report_md)
            except BenchmarkError as e:
                print(f"benchmark_qwen: report not written: {e}",
                      file=sys.stderr)
                return 2
        return 0
    except BenchmarkError as e:
        print(f"\nbenchmark_qwen: FAILED: {e}", file=sys.stderr)
        print(f"Raw artifacts preserved in: {run_dir}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
