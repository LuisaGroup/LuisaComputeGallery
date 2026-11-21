#!/usr/bin/env python3
"""A/B the stored K/V dtype of one weight format (opt_plan/05 section 4 item Q3).

The FP8 KV cache buys bandwidth and capacity but pays for an ONLINE-scale append:
every prepare call launches one extra kernel, and an append whose block maximum
grows rewrites that block's valid prefix. Whether that wins is a measurement, not
an argument, so this script runs the SAME prompt with several `--kv-dtype` values
in an interleaved order (A B C, C B A, A B C, ...) at a context long enough for the
KV read to matter, and reports the prefill (TTFT) and decode ms/token per dtype.

It drives the real CLI (no synthetic microbenchmark) so the number includes
everything the user pays: the extra append launch, the scale loads in the tiled
prefill kernel and the decode attention, and the rescale rewrites.

Usage
-----
  python scripts/benchmark_qwen_kv.py \
      --example build/windows/x64/release/example_qwen.exe --backend cuda \
      --model data/Qwen3.5-4B-BF16.gguf --kinds f32,bf16,fp8 \
      --prompt-len 8192 --decode 32 --repeats 3 --tier t16 \
      --out .kimix_cache/qwen_benchmark/phase8d/kv_ab.json

Every run is a fresh process, so the engine build (shader cache warm after the
first run) is part of what is measured; the script warms up once per dtype and
then measures `--repeats` samples, reporting the median.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

TTFT_RE = re.compile(r"TTFT\(prefill (\d+) tokens\) ([\d.]+) ms")
DECODE_RE = re.compile(r"decode ([\d.]+) ms/token \((\d+) steps\)")
COMPILE_RE = re.compile(r"engine-compile ([\d.]+) ms")
KV_RE = re.compile(r"^kv dtype: (\S+) layout=(\S+) codes/word=(\d+) payload=([\d.]+) MiB",
                   re.M)


def run_once(args, kv, prompt_len, decode):
    cmd = [args.example, "--backend", args.backend, "--model", args.model,
           "--version", args.version, "--synthetic-prompt-len", str(prompt_len),
           "--max-new-tokens", str(decode), "--temperature", "0",
           "--max-context", str(prompt_len + decode + 64), "--kv-dtype", kv]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit("%s run failed (%d):\n%s" % (kv, proc.returncode, proc.stdout[-2000:]))
    out = proc.stdout
    kv_line = KV_RE.search(out)
    ttft = TTFT_RE.search(out)
    dec = DECODE_RE.search(out)
    comp = COMPILE_RE.search(out)
    if kv_line is None or ttft is None or dec is None:
        raise SystemExit("%s run produced no timing/kv line:\n%s" % (kv, out[-2000:]))
    reported = kv_line.group(1)
    if reported != kv and not (kv == "fp8" and reported == "fp8_e4m3"):
        raise SystemExit("asked for %s but the engine built %s" % (kv, reported))
    ids = None
    for line in out.splitlines():
        if line.startswith("generated ids ("):
            ids = line.split(":", 1)[1].split()
    return {
        "kv_dtype": reported,
        "codes_per_word": int(kv_line.group(3)),
        "payload_mib": float(kv_line.group(4)),
        "prompt_tokens": int(ttft.group(1)),
        "ttft_ms": float(ttft.group(2)),
        "decode_ms_per_token": float(dec.group(1)),
        "decode_steps": int(dec.group(2)),
        "engine_compile_ms": float(comp.group(1)) if comp else None,
        "ids": ids,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="dsl-bf16")
    ap.add_argument("--kinds", default="f32,bf16,fp8")
    ap.add_argument("--prompt-len", type=int, default=8192)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tier", default="t16")
    ap.add_argument("--device-tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    if len(kinds) < 2:
        ap.error("--kinds needs at least two dtypes to A/B")
    samples = {k: [] for k in kinds}
    ids_by_kind = {}
    # Interleave the order (A B C, C B A, ...) so a thermal or cache drift cannot
    # favour one dtype systematically; the warm-up run per dtype is discarded.
    for k in kinds:
        print("warmup %s" % k, file=sys.stderr)
        run_once(args, k, args.prompt_len, args.decode)
    for rep in range(args.repeats):
        order = kinds if rep % 2 == 0 else list(reversed(kinds))
        for k in order:
            r = run_once(args, k, args.prompt_len, args.decode)
            keep = dict(r)
            keep.pop("ids")
            keep["rep"] = rep
            samples[k].append(keep)
            ids_by_kind.setdefault(k, r["ids"])
            print("%-5s rep %d: ttft %.1f ms decode %.3f ms/token (prompt %d)"
                  % (k, rep, r["ttft_ms"], r["decode_ms_per_token"], r["prompt_tokens"]),
                  file=sys.stderr)
    summary = {
        "kind": "qwen_kv_dtype_ab",
        "tier": args.tier,
        "device_tag": args.device_tag,
        "backend": args.backend,
        "model": os.path.abspath(args.model),
        "version": args.version,
        "prompt_len_requested": args.prompt_len,
        "decode_tokens": args.decode,
        "repeats": args.repeats,
        "kinds": {},
        "ids_identical_across_dtypes": len({tuple(v) for v in ids_by_kind.values()}) == 1,
    }
    for k, rows in samples.items():
        ttft = statistics.median(r["ttft_ms"] for r in rows)
        dec = statistics.median(r["decode_ms_per_token"] for r in rows)
        summary["kinds"][k] = {
            "kv_dtype_reported": rows[0]["kv_dtype"],
            "codes_per_word": rows[0]["codes_per_word"],
            "payload_mib": rows[0]["payload_mib"],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "ttft_ms_median": ttft,
            "decode_ms_per_token_median": dec,
            "engine_compile_ms_median": statistics.median(
                r["engine_compile_ms"] for r in rows if r["engine_compile_ms"] is not None),
            "samples": rows,
        }
    base = kinds[0]
    for k in kinds[1:]:
        summary["kinds"][k]["decode_speedup_vs_%s" % base] = (
            summary["kinds"][base]["decode_ms_per_token_median"] / summary["kinds"][k]["decode_ms_per_token_median"])
        summary["kinds"][k]["ttft_speedup_vs_%s" % base] = (
            summary["kinds"][base]["ttft_ms_median"] / summary["kinds"][k]["ttft_ms_median"])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                      for k, v in summary["kinds"].items()}, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
