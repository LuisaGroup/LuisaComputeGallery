#!/usr/bin/env python3
"""Quality gate for quantized Qwen3.5-4B weight formats AND KV cache dtypes.

Two gates share one analysis engine:

  --gate w4  (opt_plan/05 section 3) the W4A16 weight gate: >= 32 prompts of
             >= 128 generated tokens against the BF16 reference.
  --gate q3  (opt_plan/05 section 4 item Q3, "FP8 KV") the KV-dtype gate: the SAME
             weight format (normally dsl-bf16, so no W4 pack is involved) run with
             two DIFFERENT KV dtypes, e.g. reference `--kv-dtype f32` vs candidate
             `--kv-dtype fp8`, over >= 16 prompts at an effective 8 K context, with
             >= 98 % mean token-id agreement, KL mean/max recorded, a NaN/repeat
             check and a real needle-style long-context set (text documents with a
             passkey at a controlled depth -- NOT random operator tensors, which the
             synthetic LCG prompts are).

Every dump records `weight+kv_<dtype>` in its format field (example_qwen writes it),
so a comparison can prove both arms share the weight format and that the KV dtype
actually differs -- a mismatched or partially labelled set is REFUSED rather than
averaged. The tier and device tag are stored with the verdict exactly as before, and
a T16 result never stands in for T32.

The gate is the non-negotiable deliverable of the W4A16 item: a format is only
reported as usable when, against the BF16 reference on >= 32 fixed prompts of
>= 128 generated tokens,

  * the mean greedy prefix agreement is >= 90 %,
  * the first 32 greedy ids are identical on >= 28 of 32 prompts,
  * the mean/max KL divergence between the last-position distributions is
    recorded (KL(BF16 || candidate), with mean/max across prompts, computed
    from the FP32 logits, never from rounded text),
  * a 1 K-generation repeat/NaN check passes.

It reads the BOUNDED BINARY dumps that `example_qwen --dump-logits <file>`
writes (header + prompt ids + greedy ids + the prefill and final last-position
FP32 logits), so every number here is computed from the engine's own output, not
from a retyped table.

Two modes:

  compare  --reference <dir|file> --candidate <dir|file> [--out result.json]
           Pure analysis over stored dumps. Refuses a missing dump, a magic /
           version / vocab / prompt mismatch, a prompt-length mismatch, a tier
           mismatch (a T32 claim from a T16 run is refused), and any non-finite
           logit. Emits the metrics and the verdict.

  run      --example <exe> --backend cuda --model <bf16.gguf>
           [--w4-pack <dir>] [--reference-version dsl-bf16]
           [--candidate-version dsl-w4] [--prompts 32] [--prompt-min-len 128]
           [--max-new-tokens 128] [--generation-check 1024]
           [--tier t16|t32] [--device-tag <sku>] --out <dir>
           Drives the CLI once per (prompt, version), writes the dumps and the
           logs into <out>/, then computes the same metrics. The prompt ids are
           the harness' deterministic synthetic list (--synthetic-prompt-len),
           so the run is reproducible from this command line alone.

Every result records the tier and the device tag. A gate met on one tier says
nothing about the other; the script never merges them.

Usage examples
--------------
  python scripts/qwen_quality_gate.py run --example build/windows/x64/release/example_qwen.exe \
      --backend cuda --model data/Qwen3.5-4B-BF16.gguf \
      --w4-pack .kimix_cache/qwen_benchmark/phase8b/w4pack \
      --prompts 32 --max-new-tokens 128 --generation-check 1024 \
      --tier t16 --device-tag "RTX 4080 SUPER" \
      --out .kimix_cache/qwen_benchmark/phase8b/quality

  python scripts/qwen_quality_gate.py compare --reference dumps/bf16 --candidate dumps/w4 \
      --tier t16 --device-tag "RTX 4080 SUPER" --out summary.json
"""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys

DUMP_MAGIC = b"QWLOGITS"
DUMP_VERSION = 1
# magic, version, vocab, prompt_len, generated, temperature, pad, seed,
# format[32], backend[16], tier[8], device_tag[64]
HEADER = struct.Struct("<8sIIIIf4xQ32s16s8s64s")
assert HEADER.size == 160, HEADER.size

GATE_MEAN_PREFIX = 0.90
GATE_FIRST32_COUNT = 28
GATE_PROMPTS = 32
# Q3 KV gate (opt_plan/05 section 4): 16 prompts at 8 K effective context and 98 %
# mean token-id agreement, plus no prompt below the per-prompt floor. The needle
# set (see run_needle) is a separate, mandatory part of the same gate.
GATE_Q3_MEAN_PREFIX = 0.98
GATE_Q3_PER_PROMPT_FLOOR = 0.90
GATE_Q3_PROMPTS = 16
GATE_Q3_EFFECTIVE_CONTEXT = 8192
GATE_Q3_NEEDLE_PROMPTS = 4
KV_DTYPES = ("f32", "bf16", "fp8")


class GateError(RuntimeError):
    """Any condition that makes a metric meaningless: the script refuses."""


# ---------------------------------------------------------------------------
# Dump reader
# ---------------------------------------------------------------------------
# The dump format field is `<weight>+kv_<dtype>` since the Q3 KV work; dumps written
# before that carry only the weight part, which is reported as an UNKNOWN KV axis
# (never guessed) so a KV comparison over them is refused.
FORMAT_KV_MARKER = "+kv_"


def normalize_kv_dtype(name):
    """`fp8` (the CLI spelling) and `fp8_e4m3` (the storage name the dump records)
    are the same cache dtype."""
    if name is None:
        return None
    return "fp8_e4m3" if name == "fp8" else name


def parse_dump_format(fmt):
    """(weight_format, kv_dtype or None) from a dump format string."""
    if FORMAT_KV_MARKER in fmt:
        weight, kv = fmt.split(FORMAT_KV_MARKER, 1)
        return weight, kv
    return fmt, None


def _cstr(raw):
    end = raw.find(b"\0")
    if end < 0:
        end = len(raw)
    return raw[:end].decode("utf-8", "replace")


class Dump:
    """One example_qwen --dump-logits file."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < HEADER.size:
            raise GateError("%s: shorter than a dump header" % path)
        (magic, version, vocab, prompt_len, generated, temperature, seed,
         fmt, backend, tier, tag) = HEADER.unpack_from(blob, 0)
        if magic != DUMP_MAGIC:
            raise GateError("%s: not a qwen logits dump (magic %r)" % (path, magic))
        if version != DUMP_VERSION:
            raise GateError("%s: unsupported dump version %d" % (path, version))
        if vocab == 0 or prompt_len == 0:
            raise GateError("%s: empty vocab or prompt" % path)
        self.vocab = vocab
        self.prompt_len = prompt_len
        self.generated_count = generated
        self.temperature = temperature
        self.seed = seed
        self.format = _cstr(fmt)
        self.weight_format, self.kv_dtype = parse_dump_format(self.format)
        self.backend = _cstr(backend)
        self.tier = _cstr(tier)
        self.device_tag = _cstr(tag)
        # The final distribution is OPTIONAL: the device-greedy in-flight decode
        # path never brings the last step's logits to the host, so the CLI writes
        # only the prefill (last-prompt-position) vector then. Both layouts are
        # accepted, nothing else is.
        full = HEADER.size + 4 * (prompt_len + generated) + 8 * vocab
        prefill_only = HEADER.size + 4 * (prompt_len + generated) + 4 * vocab
        if len(blob) == full:
            self.has_final = True
        elif len(blob) == prefill_only:
            self.has_final = False
        else:
            raise GateError(
                "%s: payload is %d bytes, header geometry needs %d (prefill+final) "
                "or %d (prefill only); refusing a truncated/mixed dump"
                % (path, len(blob), full, prefill_only))
        off = HEADER.size
        self.prompt = list(struct.unpack_from("<%dI" % prompt_len, blob, off))
        off += 4 * prompt_len
        self.generated = list(struct.unpack_from("<%dI" % generated, blob, off))
        off += 4 * generated
        self.prefill_logits = list(struct.unpack_from("<%df" % vocab, blob, off))
        off += 4 * vocab
        self.final_logits = (list(struct.unpack_from("<%df" % vocab, blob, off))
                             if self.has_final else [])
        for name, vec in (("prefill", self.prefill_logits), ("final", self.final_logits)):
            for i, v in enumerate(vec):
                if not math.isfinite(v):
                    raise GateError("%s: non-finite %s logit at index %d (%r)"
                                    % (path, name, i, v))

    def key(self):
        """Identity used to pair a reference with a candidate."""
        return (self.prompt_len, self.vocab, tuple(self.prompt))


def _gather(paths):
    """Expand files and directories into a sorted list of dump paths."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".qlogits"):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise GateError("%s: no such dump file or directory" % p)
    if not files:
        raise GateError("no *.qlogits dumps found in %s" % ", ".join(paths))
    return files


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _log_softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        raise GateError("degenerate softmax (sum=%r)" % total)
    log_total = math.log(total)
    return [(v - m) - log_total for v in logits]


def kl_divergence(p_logits, q_logits):
    """KL(P || Q) in nats from two FP32 logit vectors (both renormalized)."""
    lp = _log_softmax(p_logits)
    lq = _log_softmax(q_logits)
    total = 0.0
    for a, b in zip(lp, lq):
        p = math.exp(a)
        if p > 0.0:
            total += p * (a - b)
    return total


def prefix_agreement(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compare_dumps(ref, cand):
    if (ref.prompt_len, ref.vocab) != (cand.prompt_len, cand.vocab):
        raise GateError(
            "prompt/vocab mismatch: reference prompt=%d vocab=%d, candidate prompt=%d vocab=%d"
            % (ref.prompt_len, ref.vocab, cand.prompt_len, cand.vocab))
    if ref.prompt != cand.prompt:
        raise GateError("prompt ids differ between reference and candidate; "
                        "these dumps do not describe the same experiment")
    agree = prefix_agreement(ref.generated, cand.generated)
    common = min(len(ref.generated), len(cand.generated))
    first32 = len(ref.generated) >= 32 and len(cand.generated) >= 32 and \
        ref.generated[:32] == cand.generated[:32]
    kl_prefill = kl_divergence(ref.prefill_logits, cand.prefill_logits)
    kl_final = None
    if ref.has_final and cand.has_final:
        if len(ref.final_logits) != len(cand.final_logits):
            raise GateError("final logit length mismatch")
        kl_final = kl_divergence(ref.final_logits, cand.final_logits)
    # Reference-relative agreement: how many of the candidate's ids match the
    # reference at the same position, over the reference length.
    matched = sum(1 for i in range(common) if ref.generated[i] == cand.generated[i])
    return {
        "prompt_len": ref.prompt_len,
        "generated": common,
        "prefix_agreement_ids": agree,
        "prefix_agreement": (agree / common) if common else 0.0,
        "matched_over_reference": (matched / len(ref.generated)) if ref.generated else 0.0,
        "identical_first_32": bool(first32),
        "kl_prefill": kl_prefill,
        "kl_final": kl_final,
        "reference_ids": ref.generated[:64],
        "candidate_ids": cand.generated[:64],
        "reference_format": ref.format,
        "candidate_format": cand.format,
        "reference_backend": ref.backend,
        "candidate_backend": cand.backend,
    }


def evaluate(pairs, tier, device_tag, require_full=False, gate="w4", needle=None,
             reference_kv_dtype=None, candidate_kv_dtype=None):
    """Turn paired dumps into the gate verdict. `pairs` is [(ref, cand)].

    `gate` selects the threshold family: "w4" is the opt_plan/05 section 3 weight
    gate, "q3" the section 4 KV-dtype gate. `needle` is the needle-set summary
    (run_needle's return value) and is REQUIRED for a q3 pass -- a long-context
    retrieval set is the whole point of the Q3 gate, so a run without it cannot
    claim support.
    """
    if not pairs:
        raise GateError("no prompt dumps to compare")
    metrics = []
    for ref, cand in pairs:
        metrics.append(compare_dumps(ref, cand))
    tiers = sorted({ref.tier for ref, _ in pairs} | {cand.tier for _, cand in pairs})
    tags = sorted({ref.device_tag for ref, _ in pairs} | {cand.device_tag for _, cand in pairs})
    tiers = [t for t in tiers if t]
    if len(tiers) > 1:
        raise GateError("dumps span several tiers %s; a gate is per tier" % tiers)
    if tier is None:
        tier = tiers[0] if tiers else ""
    elif tiers and tiers[0] != tier:
        raise GateError("dump tier %r does not match the requested tier %r; "
                        "a T32 gate cannot be claimed from a T16 run" % (tiers[0], tier))
    if device_tag is None:
        device_tag = ", ".join(t for t in tags if t)
    elif tags and any(t and device_tag not in t and t not in device_tag for t in tags):
        # Identical-or-containing is tolerated (a SKU vs a longer tag), anything
        # else means the dumps came from another device.
        raise GateError("dump device tags %s do not match --device-tag %r" % (tags, device_tag))
    n = len(metrics)
    mean_prefix = sum(m["prefix_agreement"] for m in metrics) / n
    first32_ok = sum(1 for m in metrics if m["identical_first_32"])
    kl_prefill = [m["kl_prefill"] for m in metrics]
    kl_final = [m["kl_final"] for m in metrics if m["kl_final"] is not None]
    worst_prefix = min(m["prefix_agreement"] for m in metrics)
    # Both axes of every dump must be consistent with the gate being claimed: the
    # weight format must be the SAME on both arms (the Q3 gate varies the KV dtype
    # only), and a q3 run must show the requested KV dtypes. Anything else is a
    # mismatched set and is refused instead of being averaged into a verdict.
    ref_formats = sorted({m["reference_format"] for m in metrics})
    cand_formats = sorted({m["candidate_format"] for m in metrics})
    weights = sorted({parse_dump_format(f)[0] for f in ref_formats + cand_formats})
    ref_kv = sorted({parse_dump_format(f)[1] for f in ref_formats}, key=str)
    cand_kv = sorted({parse_dump_format(f)[1] for f in cand_formats}, key=str)
    if gate == "q3":
        # The Q3 gate varies the KV dtype ONLY: both arms must share one weight
        # format, otherwise the verdict confounds two effects at once.
        if len(weights) > 1:
            raise GateError("the q3 dumps mix several WEIGHT formats %s; this gate "
                            "compares two KV dtypes of ONE weight format" % weights)
        if None in ref_kv or None in cand_kv:
            raise GateError("a q3 comparison needs the KV dtype recorded in every dump "
                            "(format 'weight+kv_<dtype>'); these dumps only carry %s/%s"
                            % (ref_formats, cand_formats))
        if len(ref_kv) != 1 or len(cand_kv) != 1:
            raise GateError("the dumps mix KV dtypes on one arm: reference %s, candidate %s"
                            % (ref_kv, cand_kv))
        if normalize_kv_dtype(ref_kv[0]) == normalize_kv_dtype(cand_kv[0]):
            raise GateError("both arms use KV dtype %r; a q3 gate compares two dtypes"
                            % ref_kv[0])
        for side, got, want in (("reference", ref_kv[0], reference_kv_dtype),
                                ("candidate", cand_kv[0], candidate_kv_dtype)):
            if want and normalize_kv_dtype(got) != normalize_kv_dtype(want):
                raise GateError("the %s dumps were produced with KV dtype %r but %r was "
                                "requested; refusing to score a different experiment"
                                % (side, got, want))
    gate_prompts = GATE_Q3_PROMPTS if gate == "q3" else GATE_PROMPTS
    mean_threshold = GATE_Q3_MEAN_PREFIX if gate == "q3" else GATE_MEAN_PREFIX
    if gate == "q3":
        prompt_ok = mean_prefix >= GATE_Q3_MEAN_PREFIX and \
            worst_prefix >= GATE_Q3_PER_PROMPT_FLOOR
    else:
        prompt_ok = mean_prefix >= GATE_MEAN_PREFIX and first32_ok >= GATE_FIRST32_COUNT
    prompt_gate = "insufficient_prompts" if n < gate_prompts else (
        "pass" if prompt_ok else "fail")
    if require_full and n < gate_prompts:
        prompt_gate = "insufficient_prompts"
    # The needle set is MANDATORY for a q3 pass: a long-context retrieval set is
    # the only evidence that the extra KV quantization does not destroy recall.
    needle_gate = "not_measured"
    if gate == "q3":
        if needle is None:
            needle_gate = "missing"
        else:
            needle_gate = needle["verdict"]
    verdict = "not_measured" if n < gate_prompts else ("pass" if prompt_gate == "pass" else "fail")
    if gate == "q3" and verdict == "pass" and needle_gate != "pass":
        verdict = "fail"
    summary = {
        "kind": "qwen_quality_gate",
        "version": 1,
        "gate_name": gate,
        "tier": tier,
        "device_tag": device_tag,
        "prompts": n,
        "weight_format": weights[0] if weights else None,
        "kv_dtype": {"reference": ref_kv[0] if len(ref_kv) == 1 else ref_kv,
                     "candidate": cand_kv[0] if len(cand_kv) == 1 else cand_kv},
        "gate": {
            "mean_prefix_agreement_threshold": mean_threshold,
            "first32_identical_threshold": GATE_FIRST32_COUNT,
            "per_prompt_floor": GATE_Q3_PER_PROMPT_FLOOR if gate == "q3" else None,
            "prompts_required": gate_prompts,
            "needle_gate": needle_gate,
            "verdict": verdict,
            "prompt_gate": prompt_gate,
        },
        "mean_prefix_agreement": mean_prefix,
        "worst_prompt_agreement": worst_prefix,
        "first32_identical": first32_ok,
        "kl_prefill": {
            "mean": sum(kl_prefill) / n,
            "max": max(kl_prefill),
        },
        # The final last-position distribution is only present when the run used
        # the host-logits path (--decode-flight 0): the device-greedy in-flight
        # decode never brings it back, and the CLI then writes the prefill vector
        # only. A missing final vector is reported as null, never fabricated.
        "kl_final": ({
            "mean": sum(kl_final) / len(kl_final),
            "max": max(kl_final),
            "prompts": len(kl_final),
        } if kl_final else None),
        "per_prompt": metrics,
        "needle": needle,
    }
    return summary


# ---------------------------------------------------------------------------
# Synthetic prompt ids (must match example_qwen::qwen_synthetic_prompt_ids and
# scripts/benchmark_qwen.py; the CLI regenerates them, so only the LENGTH is
# passed on the command line).
# ---------------------------------------------------------------------------
PROMPT_ID_MIN = 1000
PROMPT_ID_MAX = 200000
_PROMPT_SEED = 0x0C0FFEE5EEDBEEF01
_PROMPT_LENGTH_MIX = 0xD1B54A32D192ED03
_PROMPT_LCG_MUL = 6364136223846793005
_PROMPT_LCG_INC = 1442695040888963407
_U64_MASK = (1 << 64) - 1


def synthetic_prompt_ids(length):
    """Same rule as qwen/prompt_ids.h and scripts/benchmark_qwen.py."""
    if length < 1:
        raise GateError("prompt length must be >= 1, got %d" % length)
    span = PROMPT_ID_MAX - PROMPT_ID_MIN
    state = (_PROMPT_SEED ^ ((length * _PROMPT_LENGTH_MIX) & _U64_MASK)) & _U64_MASK
    ids = []
    for _ in range(length):
        state = (state * _PROMPT_LCG_MUL + _PROMPT_LCG_INC) & _U64_MASK
        ids.append(PROMPT_ID_MIN + ((state >> 32) % span))
    return ids


# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------
# Default prompt-length cycle of the 32-prompt gate. The plan asks for >= 128
# tokens per prompt; the cycle stops at 4096 because a longer prompt needs a
# bigger KV capacity (--max-context) and multiplies the run time without adding
# quality information. Pass --prompt-lengths to override.
PROMPT_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
ID_LINE = re.compile(r"^generated ids \((\d+)\):(.*)$")


def _parse_ids(stdout):
    """The CLI's one-line `generated ids (N): ...` record, or None."""
    found = None
    for line in stdout.splitlines():
        m = ID_LINE.match(line.strip())
        if m:
            ids = [int(t) for t in m.group(2).split()] if m.group(2).strip() else []
            if int(m.group(1)) != len(ids):
                raise GateError("generated-id line declares %s ids but carries %d"
                                % (m.group(1), len(ids)))
            found = ids
    return found


KV_LINE = re.compile(r"^kv dtype: (\S+) layout", re.M)
TTFT_LINE = re.compile(r"TTFT\(prefill (\d+) tokens\)")
# The generated text may contain newlines (the model often emits blank lines
# before the answer), so it runs from the "generated text:" line to the "stop:"
# line that always follows it.
TEXT_LINE = re.compile(r"^generated text: (.*?)^stop:", re.M | re.S)


def _run_one(example, backend, model, version, extra, prompt_len, max_new_tokens,
             dump_path, log_path, tier, device_tag, max_context=0,
             kv_dtype=None, prompt_file=None):
    """One example_qwen run. Returns (ids, log text). Refuses a run whose log does
    not report the KV dtype it was asked for -- the KV axis is what the q3 gate
    compares, so an unverified dtype would make the whole comparison meaningless."""
    cmd = [example, "--backend", backend, "--model", model, "--version", version,
           "--max-new-tokens", str(max_new_tokens), "--temperature", "0",
           "--dump-logits", dump_path, "--tier", tier]
    if prompt_file is None:
        cmd += ["--synthetic-prompt-len", str(prompt_len)]
    else:
        cmd += ["--prompt-file", prompt_file]
    if kv_dtype:
        cmd += ["--kv-dtype", kv_dtype]
    if max_context:
        cmd += ["--max-context", str(max_context)]
    if device_tag:
        cmd += ["--device-tag", device_tag]
    cmd += extra
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        log.write(proc.stdout)
    if proc.returncode != 0:
        raise GateError("%s %s exited %d (see %s)"
                        % (version, os.path.basename(dump_path), proc.returncode, log_path))
    ids = _parse_ids(proc.stdout)
    if ids is None:
        raise GateError("%s produced no 'generated ids' line (see %s)" % (version, log_path))
    if not os.path.isfile(dump_path):
        raise GateError("%s did not write the requested dump %s" % (version, dump_path))
    if kv_dtype:
        m = KV_LINE.search(proc.stdout)
        if m is None:
            raise GateError("%s did not report its KV dtype (see %s)" % (version, log_path))
        reported = m.group(1)
        # `fp8` and `fp8_e4m3` name the same storage; everything else must match.
        if reported != kv_dtype and not (kv_dtype == "fp8" and reported == "fp8_e4m3"):
            raise GateError("%s ran with KV dtype %r but %r was requested (see %s)"
                            % (version, reported, kv_dtype, log_path))
    return ids, proc.stdout


# ---------------------------------------------------------------------------
# Needle set: REAL TEXT at a controlled depth, not random operator tensors.
#
# The synthetic LCG prompts exercise the kernels but say nothing about long-context
# RETRIEVAL, which is exactly what a coarser KV dtype threatens (the plan is
# explicit: "long-context retrieval must be tested, not just the attention kernel
# error"). Each document is a deterministic filler narrative with one passkey
# sentence inserted at a depth fraction, followed by a question. The gate requires
# the REFERENCE arm to retrieve the passkey (otherwise the fixture is broken, not
# the candidate) and the CANDIDATE arm to retrieve it too, and records the token
# agreement on the same prompt.
# ---------------------------------------------------------------------------
NEEDLE_FILLER = (
    "The harbor office keeps a ledger of every ship that enters the bay. ",
    "Each entry records the cargo, the crew size and the day of arrival. ",
    "Weather in the strait changes quickly, so the pilots file a new report hourly. ",
    "The lighthouse keeper notes the visibility and the direction of the swell. ",
    "Merchants in the old quarter compare prices before the market opens. ",
    "A steady wind from the north usually means the ferry will be late. ",
    "The archive stores the ledgers in wooden cabinets along the western wall. ",
    "Scribes copy damaged pages by hand and mark the repairs in red ink. ",
    "Fishermen repair their nets on the quay while the tide is low. ",
    "The custom house clock is five minutes ahead of the church bell. ",
)
NEEDLE_TEMPLATE = "The pass key is {key}. Remember it exactly.\n"
NEEDLE_QUESTION = "\n\nQuestion: What is the pass key? Answer with the digits only.\n"


def build_needle_document(ctx_tokens, passkey, depth, chars_per_token=4.0):
    """A deterministic document of roughly `ctx_tokens` tokens with one passkey.

    The exact prompt length is MEASURED by the CLI (its `TTFT(prefill N tokens)`
    line) and stored with the result, so the reported context is never an estimate.
    """
    if not 0.0 < depth < 1.0:
        raise GateError("--needle-depth must be in (0, 1), got %r" % depth)
    target_chars = max(256, int(ctx_tokens * chars_per_token))
    filler = []
    total = 0
    i = 0
    while total < target_chars:
        piece = NEEDLE_FILLER[i % len(NEEDLE_FILLER)]
        filler.append(piece)
        total += len(piece)
        i += 1
    text = "".join(filler)
    cut = int(len(text) * depth)
    space = text.find(" ", cut)
    cut = len(text) if space < 0 else space + 1
    key_line = NEEDLE_TEMPLATE.replace("{key}", passkey)
    return text[:cut] + key_line + text[cut:] + NEEDLE_QUESTION


DIGITS_ONLY = re.compile(r"[^0-9]")


def _digits_only(text):
    """The passkey is compared against the digits of the generated text, so a
    formatted answer ("1,000,000") still counts as a retrieval."""
    return DIGITS_ONLY.sub("", text)


def _generated_text(stdout):
    m = TEXT_LINE.search(stdout)
    return (m.group(1) if m else "").strip()


def _prompt_tokens(stdout):
    m = TTFT_LINE.search(stdout)
    return int(m.group(1)) if m else None


def run_needle(args, out_dir, candidate_extra):
    """The needle-style long-context set for the q3 gate.

    The document length is CALIBRATED against the tokenizer: the first document is
    built with a guessed characters-per-token ratio, the CLI reports the real prompt
    length (`TTFT(prefill N tokens)`) and the ratio is corrected until the measured
    length reaches the requested context. The measured count is what the summary
    reports, so an "8 K context" claim is never an estimate.
    """
    docs = os.path.join(out_dir, "needle")
    logs = os.path.join(docs, "logs")
    os.makedirs(logs, exist_ok=True)
    ratio = 4.0
    results = []
    for i in range(args.needle_prompts):
        passkey = "%07d" % (1000000 + (i * 7919) % 8999999)
        depth = 0.1 + 0.8 * ((i * 37) % 100) / 100.0
        doc = os.path.join(docs, "doc%02d.txt" % i)
        tokens_by_arm = {}
        ids_by_arm = {}
        text_by_arm = {}
        arms = (("reference", args.reference_version, args.reference_kv_dtype,
                 args.reference_extra),
                ("candidate", args.candidate_version, args.candidate_kv_dtype,
                 candidate_extra))
        # Attempt 1 calibrates the length from the REFERENCE arm only; a second
        # attempt runs both arms on the corrected document.
        for attempt in (0, 1):
            text = build_needle_document(args.needle_ctx, passkey, depth, ratio)
            with open(doc, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            ids_by_arm.clear()
            text_by_arm.clear()
            tokens_by_arm.clear()
            for arm, version, kv, extra in arms:
                dump = os.path.join(docs, "doc%02d.%s.qlogits" % (i, arm))
                log = os.path.join(logs, "doc%02d.%s.log" % (i, arm))
                run_extra = list(extra) + ["--chat"]
                ids, stdout = _run_one(args.example, args.backend, args.model, version,
                                       run_extra, 0, args.needle_max_new_tokens, dump, log,
                                       args.tier, args.device_tag, args.needle_max_context,
                                       kv_dtype=kv, prompt_file=doc)
                ids_by_arm[arm] = ids
                text_by_arm[arm] = _generated_text(stdout)
                tokens_by_arm[arm] = _prompt_tokens(stdout)
                if arm == "reference":
                    # Calibrate from the measured length of THIS document.
                    measured = tokens_by_arm[arm]
                    if measured and measured > 0:
                        ratio = max(1.0, len(text) / float(measured))
            measured = tokens_by_arm.get("reference") or 0
            if measured >= int(args.needle_ctx * 0.98):
                break
            if attempt == 1:
                print("warn: needle doc %d measured %d tokens (< %d requested); "
                      "recorded as measured" % (i, measured, args.needle_ctx),
                      file=sys.stderr)
        entry = {"prompt": i, "passkey": passkey, "depth": round(depth, 3),
                 "doc": os.path.abspath(doc), "doc_bytes": os.path.getsize(doc)}
        common = min(len(ids_by_arm["reference"]), len(ids_by_arm["candidate"]))
        agree = sum(1 for k in range(common)
                    if ids_by_arm["reference"][k] == ids_by_arm["candidate"][k])
        ref_hit = entry["passkey"] in _digits_only(text_by_arm["reference"])
        cand_hit = entry["passkey"] in _digits_only(text_by_arm["candidate"])
        entry.update({
            "prompt_tokens": tokens_by_arm["reference"],
            "reference_hit": ref_hit,
            "candidate_hit": cand_hit,
            "ids_agreement": (agree / common) if common else 0.0,
            "reference_text": text_by_arm["reference"][:200],
            "candidate_text": text_by_arm["candidate"][:200],
        })
        results.append(entry)
        print("needle %d/%d depth=%.2f key=%s ref_hit=%s cand_hit=%s agreement=%.3f"
              % (i + 1, args.needle_prompts, entry["depth"], passkey, ref_hit, cand_hit,
                 entry["ids_agreement"]), file=sys.stderr)
    hits = sum(1 for r in results if r["candidate_hit"])
    ref_hits = sum(1 for r in results if r["reference_hit"])
    # The gate is "NO DEGRADATION on a needle-style prompt set" (opt_plan/05 section
    # 4), not "the 4B model must answer every document": the reference's own hit
    # count is the bar, and a fixture the reference cannot solve at all proves
    # nothing, so it is refused as "invalid_fixture" instead of being scored.
    if ref_hits == 0:
        verdict = "invalid_fixture"
    elif hits >= ref_hits:
        verdict = "pass"
    else:
        verdict = "fail"
    return {
        "kind": "needle_long_context",
        "prompts": len(results),
        "context_tokens_requested": args.needle_ctx,
        "max_new_tokens": args.needle_max_new_tokens,
        "reference_hits": ref_hits,
        "candidate_hits": hits,
        "mean_ids_agreement": (sum(r["ids_agreement"] for r in results) / len(results))
        if results else 0.0,
        "verdict": verdict,
        "per_prompt": results,
    }


def run_mode(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    dumps = os.path.join(out, "dumps")
    logs = os.path.join(out, "logs")
    os.makedirs(dumps, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    if args.gate == "q3":
        # The Q3 gate's own defaults: 16 prompts, every one at an effective 8 K
        # context (prompt + generated >= 8192) unless the caller overrides them.
        if args.prompts == GATE_PROMPTS:
            args.prompts = GATE_Q3_PROMPTS
        if args.prompt_lengths is None:
            args.prompt_lengths = ",".join([str(GATE_Q3_EFFECTIVE_CONTEXT)] * args.prompts)
        if args.max_new_tokens == 128:
            args.max_new_tokens = 64
        if args.prompt_min_len == 128:
            args.prompt_min_len = GATE_Q3_EFFECTIVE_CONTEXT - args.max_new_tokens
        if args.max_context == 0:
            args.max_context = GATE_Q3_EFFECTIVE_CONTEXT + args.max_new_tokens
        if args.generation_check == 0:
            args.generation_check = 64
    lengths = [PROMPT_LENGTHS[i % len(PROMPT_LENGTHS)] for i in range(args.prompts)]
    if args.prompt_lengths:
        lengths = [int(x) for x in args.prompt_lengths.split(",")]
        if len(lengths) != args.prompts:
            raise GateError("--prompt-lengths must list exactly --prompts values")
    if min(lengths) < args.prompt_min_len:
        raise GateError("every prompt length must be >= %d (plan requirement)" % args.prompt_min_len)
    if args.gate == "q3":
        effective = min(lengths) + args.max_new_tokens
        if effective < GATE_Q3_EFFECTIVE_CONTEXT:
            raise GateError("the q3 gate needs an effective context of %d tokens "
                            "(shortest prompt %d + %d generated); refusing to report a "
                            "long-context verdict from a shorter run"
                            % (GATE_Q3_EFFECTIVE_CONTEXT, min(lengths), args.max_new_tokens))
    candidate_extra = []
    if args.candidate_version == "dsl-w4":
        if not args.w4_pack:
            raise GateError("--candidate-version dsl-w4 requires --w4-pack")
        candidate_extra = ["--w4-pack", args.w4_pack]
    pairs = []
    skipped = 0
    for i, length in enumerate(lengths):
        name = "p%02d_P%d" % (i, length)
        ref_dump = os.path.join(dumps, name + ".ref.qlogits")
        cand_dump = os.path.join(dumps, name + ".cand.qlogits")
        ref_ids, _ = _run_one(args.example, args.backend, args.model,
                              args.reference_version, args.reference_extra,
                              length, args.max_new_tokens, ref_dump,
                              os.path.join(logs, name + ".ref.log"), args.tier,
                              args.device_tag, args.max_context,
                              kv_dtype=args.reference_kv_dtype)
        try:
            cand_ids, _ = _run_one(args.example, args.backend, args.model,
                                   args.candidate_version, candidate_extra,
                                   length, args.max_new_tokens, cand_dump,
                                   os.path.join(logs, name + ".cand.log"), args.tier,
                                   args.device_tag, args.max_context,
                                   kv_dtype=args.candidate_kv_dtype)
        except GateError as e:
            # A candidate refusing a prompt (e.g. a capacity limit) is recorded,
            # never silently dropped from the denominator.
            print("warn: %s" % e, file=sys.stderr)
            skipped += 1
            continue
        pairs.append((Dump(ref_dump), Dump(cand_dump)))
    needle = None
    if args.needle_prompts > 0:
        print("needle set: %d documents x ~%d tokens" % (args.needle_prompts, args.needle_ctx),
              file=sys.stderr)
        needle = run_needle(args, out, candidate_extra)
    summary = evaluate(pairs, args.tier, args.device_tag, require_full=True, gate=args.gate,
                       needle=needle,
                       reference_kv_dtype=args.reference_kv_dtype,
                       candidate_kv_dtype=args.candidate_kv_dtype)
    summary["run"] = {
        "example": os.path.abspath(args.example),
        "backend": args.backend,
        "model": os.path.abspath(args.model),
        "w4_pack": os.path.abspath(args.w4_pack) if args.w4_pack else None,
        "reference_version": args.reference_version,
        "candidate_version": args.candidate_version,
        "reference_kv_dtype": args.reference_kv_dtype,
        "candidate_kv_dtype": args.candidate_kv_dtype,
        "prompt_lengths": lengths,
        "max_new_tokens": args.max_new_tokens,
        "effective_context": min(lengths) + args.max_new_tokens,
        "skipped_prompts": skipped,
        "needle_prompts": args.needle_prompts,
    }
    # 1 K-generation repeat/NaN check: two identical long runs must produce the
    # same ids, and the dumps must be finite (the reader already refuses NaN/inf).
    if args.generation_check > 0:
        print("generation check: %d tokens x 2 repeats" % args.generation_check, file=sys.stderr)
        # The repeat/NaN check must run on a prompt whose text is not already
        # degenerate: the synthetic LCG ids make this model emit one repeated digit,
        # which `_degenerate` correctly rejects -- but that would measure the fixture,
        # not the cache. The q3 gate therefore repeats the check on a REAL needle
        # document (the first one), which is also a longer, harder generation.
        gen_prompt_file = None
        gen_prompt_len = args.generation_check
        if needle is not None and needle["per_prompt"]:
            gen_prompt_file = needle["per_prompt"][0]["doc"]
        reps = []
        for rep in range(2):
            path = os.path.join(dumps, "gencheck_r%d.qlogits" % rep)
            ids, _ = _run_one(args.example, args.backend, args.model,
                              args.candidate_version, candidate_extra,
                              gen_prompt_len, args.generation_check, path,
                              os.path.join(logs, "gencheck_r%d.log" % rep),
                              args.tier, args.device_tag,
                              args.needle_max_context if gen_prompt_file else args.max_context,
                              kv_dtype=args.candidate_kv_dtype,
                              prompt_file=gen_prompt_file)
            d = Dump(path)
            if not d.generated or len(d.generated) < args.generation_check:
                raise GateError("generation check produced %d ids, wanted %d"
                                % (len(d.generated), args.generation_check))
            reps.append(ids)
    summary["long_generation"] = {
        "tokens_requested": args.generation_check,
        "prompt": gen_prompt_file or "synthetic-lcg",
        "ids_identical": reps[0] == reps[1],
        "nan_or_inf_logits": False,
        "signal": _repeat_signal(reps[0]),
        "verdict": "pass" if (reps[0] == reps[1] and not _degenerate(reps[0])) else "fail",
    }
    # Repeat/NaN is part of BOTH gate definitions, not an informational side
    # table. A prompt/needle pass with a degenerate or non-reproducible long
    # generation must not leave the overall verdict green.
    long_ok = (reps[0] == reps[1] and not _degenerate(reps[0]))
    summary["gate"]["long_generation_gate"] = "pass" if long_ok else "fail"
    if not long_ok:
        summary["gate"]["verdict"] = "fail"
    with open(args.out_json or os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(_render_table(summary))
    return summary


def _repeat_signal(ids):
    """Cheap repetition diagnostics: longest run of one id, and the top-3 share."""
    longest = 1
    run = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    counts = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    share = sorted((c / len(ids) for c in counts.values()), reverse=True)[:3]
    return {"longest_single_id_run": longest, "top3_id_share": share}


def _degenerate(ids):
    """Reject a 1 K generation that collapses to a short periodic cycle.

    The old all-one-id check missed an equally unusable exact cycle such as
    [220, 17, 15, 16, 22] repeated forever. A period <= 64 explaining >= 95 %
    of the sequence is a repeat failure; ordinary text with the same small
    vocabulary but no persistent period is not rejected by this heuristic.
    """
    if not ids:
        return True
    if len(set(ids)) <= 1:
        return True
    # Two ids alternating with jitter are still a repeat collapse even when no
    # exact short period exists. The long W4 run is a concrete example: its top
    # two ids occupy 84 % of 1024 tokens while longest_single_id_run is only 1.
    counts = {}
    for token in ids:
        counts[token] = counts.get(token, 0) + 1
    shares = sorted((count / len(ids) for count in counts.values()), reverse=True)
    if len(shares) >= 2 and shares[0] + shares[1] >= 0.80:
        return True
    n = len(ids)
    for period in range(1, min(64, n // 4) + 1):
        matches = sum(ids[i] == ids[i % period] for i in range(period, n))
        if matches >= 0.95 * (n - period):
            return True
    return False


def _render_table(summary):
    gate = summary["gate"]
    kv = summary.get("kv_dtype") or {}
    lines = [
        "gate=%s weight=%s kv=%s->%s" % (summary.get("gate_name", "?"),
                                         summary.get("weight_format") or "?",
                                         kv.get("reference") or "?",
                                         kv.get("candidate") or "?"),
        "tier=%s device=%s prompts=%d" % (summary["tier"] or "?", summary["device_tag"] or "?", summary["prompts"]),
        "mean prefix agreement: %.4f (threshold %.2f)"
        % (summary["mean_prefix_agreement"], gate["mean_prefix_agreement_threshold"]),
        "worst prompt agreement: %.4f" % summary["worst_prompt_agreement"],
    ] + ([
        "first-32 identical: %d/%d (threshold %d)"
        % (summary["first32_identical"], summary["prompts"], gate["first32_identical_threshold"]),
    ] if summary.get("gate_name", "w4") == "w4" else []) + [
        "KL prefill: mean %.5f max %.5f" % (summary["kl_prefill"]["mean"], summary["kl_prefill"]["max"]),
        ("KL final:   mean %.5f max %.5f" % (summary["kl_final"]["mean"], summary["kl_final"]["max"])
         if summary["kl_final"] else "KL final:   n/a (device-greedy in-flight decode returns ids only)"),
    ]
    if summary.get("needle"):
        n = summary["needle"]
        lines.append("needle: %d hits (reference %d, candidate %d) contexts=%s mean_agreement=%.3f verdict=%s"
                     % (n["candidate_hits"], n["reference_hits"], n["candidate_hits"],
                        [p.get("prompt_tokens") for p in n["per_prompt"]],
                        n["mean_ids_agreement"], n["verdict"]))
    lines.append("verdict: %s (%s)" % (gate["verdict"], gate["prompt_gate"]))
    if "long_generation" in summary:
        lg = summary["long_generation"]
        lines.append("long generation: %d tokens identical=%s verdict=%s"
                     % (lg["tokens_requested"], lg["ids_identical"], lg["verdict"]))
    return "\n".join(lines)


def needle_mode(args):
    """Run only the needle-style set (used to refresh the retrieval evidence of an
    existing dump pair without re-running the 16-prompt sweep)."""
    os.makedirs(args.out, exist_ok=True)
    summary = run_needle(args, args.out, list(args.candidate_extra or []))
    path = os.path.join(args.out, "needle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2))
    print("needle summary: %s" % path)
    return summary


def compare_mode(args):
    refs = [Dump(p) for p in _gather(args.reference)]
    cands = [Dump(p) for p in _gather(args.candidate)]
    by_key = {}
    for d in refs:
        by_key.setdefault(d.key(), []).append(d)
    pairs = []
    used = set()
    for c in cands:
        bucket = by_key.get(c.key())
        if not bucket:
            raise GateError("%s: no reference dump with the same prompt/vocab" % c.path)
        ref = bucket.pop(0)
        used.add(ref.path)
        pairs.append((ref, c))
    missing = [p for p in refs if p.path not in used]
    if missing:
        raise GateError("reference dumps without a candidate: %s" % ", ".join(missing))
    needle = None
    if getattr(args, "needle_json", None):
        with open(args.needle_json, encoding="utf-8") as f:
            needle = json.load(f)
        if needle.get("kind") != "needle_long_context":
            raise GateError("%s is not a needle summary" % args.needle_json)
    summary = evaluate(pairs, args.tier, args.device_tag,
                       require_full=args.require_full_prompt_count, gate=args.gate,
                       needle=needle,
                       reference_kv_dtype=args.reference_kv_dtype,
                       candidate_kv_dtype=args.candidate_kv_dtype)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    print(_render_table(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("compare", help="analyze stored dumps")
    c.add_argument("--reference", nargs="+", required=True)
    c.add_argument("--candidate", nargs="+", required=True)
    c.add_argument("--tier", choices=["t16", "t32"], default=None)
    c.add_argument("--device-tag", default=None)
    c.add_argument("--require-full-prompt-count", action="store_true")
    c.add_argument("--gate", choices=["w4", "q3"], default="w4")
    c.add_argument("--reference-kv-dtype", choices=list(KV_DTYPES), default=None)
    c.add_argument("--candidate-kv-dtype", choices=list(KV_DTYPES), default=None)
    c.add_argument("--needle-json", default=None,
                   help="a needle summary produced by the `needle` subcommand; without "
                        "it a q3 verdict cannot pass")
    c.add_argument("--out", default=None)

    r = sub.add_parser("run", help="drive example_qwen and analyze the dumps")
    r.add_argument("--example", required=True)
    r.add_argument("--backend", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--w4-pack", default=None)
    r.add_argument("--gate", choices=["w4", "q3"], default="w4")
    r.add_argument("--reference-version", default="dsl-bf16")
    r.add_argument("--candidate-version", default="dsl-w4")
    r.add_argument("--reference-kv-dtype", choices=list(KV_DTYPES), default=None,
                   help="example_qwen --kv-dtype for the REFERENCE arm (q3: f32)")
    r.add_argument("--candidate-kv-dtype", choices=list(KV_DTYPES), default=None,
                   help="example_qwen --kv-dtype for the CANDIDATE arm (q3: fp8)")
    r.add_argument("--needle-prompts", type=int, default=0,
                   help="needle-style long-context documents to run (q3 needs >= 1)")
    r.add_argument("--needle-ctx", type=int, default=GATE_Q3_EFFECTIVE_CONTEXT,
                   help="approximate document length in tokens (measured from the log)")
    r.add_argument("--needle-max-new-tokens", type=int, default=64)
    r.add_argument("--needle-max-context", type=int, default=0,
                   help="KV capacity for the needle runs (0 = needle_ctx x 1.5)")
    r.add_argument("--reference-extra", nargs=argparse.REMAINDER, default=[])
    r.add_argument("--prompts", type=int, default=GATE_PROMPTS)
    r.add_argument("--prompt-min-len", type=int, default=128)
    r.add_argument("--prompt-lengths", default=None)
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument("--generation-check", type=int, default=0)
    r.add_argument("--max-context", type=int, default=0,
                   help="example_qwen --max-context for every run (KV capacity); 0 keeps "
                        "the CLI default. Must exceed prompt + max-new-tokens")
    r.add_argument("--tier", choices=["t16", "t32"], required=True)
    r.add_argument("--device-tag", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--out-json", default=None)

    n = sub.add_parser("needle", help="run only the needle-style long-context set")
    n.add_argument("--example", required=True)
    n.add_argument("--backend", required=True)
    n.add_argument("--model", required=True)
    n.add_argument("--reference-version", default="dsl-bf16")
    n.add_argument("--candidate-version", default="dsl-bf16")
    n.add_argument("--reference-kv-dtype", choices=list(KV_DTYPES), default="f32")
    n.add_argument("--candidate-kv-dtype", choices=list(KV_DTYPES), default="fp8")
    n.add_argument("--reference-extra", nargs=argparse.REMAINDER, default=[])
    n.add_argument("--candidate-extra", nargs=argparse.REMAINDER, default=[])
    n.add_argument("--needle-prompts", type=int, default=GATE_Q3_NEEDLE_PROMPTS)
    n.add_argument("--needle-ctx", type=int, default=GATE_Q3_EFFECTIVE_CONTEXT)
    n.add_argument("--needle-max-new-tokens", type=int, default=64)
    n.add_argument("--needle-max-context", type=int, default=0)
    n.add_argument("--tier", choices=["t16", "t32"], required=True)
    n.add_argument("--device-tag", default="")
    n.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.mode == "needle":
        if args.needle_max_context <= 0:
            args.needle_max_context = int(args.needle_ctx * 1.5) + args.needle_max_new_tokens
        try:
            summary = needle_mode(args)
        except GateError as e:
            print("qwen_quality_gate: REFUSED: %s" % e, file=sys.stderr)
            return 2
        return 0 if summary["verdict"] == "pass" else 3
    if args.mode == "run" and args.gate == "q3" and args.needle_prompts <= 0:
        ap.error("--gate q3 requires --needle-prompts >= 1: a long-context gate without "
                 "a retrieval set cannot claim support")
    if args.mode == "run" and args.gate == "q3":
        # The Q3 gate compares two KV dtypes of ONE weight format, so neither arm
        # needs a W4 pack and both default to the plain BF16 weight format.
        if args.candidate_version == "dsl-w4":
            args.candidate_version = "dsl-bf16"
        if args.reference_kv_dtype is None:
            args.reference_kv_dtype = "f32"
        if args.candidate_kv_dtype is None:
            args.candidate_kv_dtype = "fp8"
        if args.needle_max_context <= 0:
            args.needle_max_context = int(args.needle_ctx * 1.5) + args.needle_max_new_tokens
    try:
        if args.mode == "compare":
            summary = compare_mode(args)
            if summary["gate"]["verdict"] != "pass":
                return 3
        else:
            summary = run_mode(args)
            gate = summary["gate"]
            # Exit code: 0 only for a MEASURED pass; a partial measurement or a
            # failed gate is nonzero so a CI/harness cannot mistake it for a pass.
            if gate["verdict"] != "pass":
                return 3
    except GateError as e:
        print("qwen_quality_gate: REFUSED: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
