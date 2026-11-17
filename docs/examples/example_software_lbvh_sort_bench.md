# example_software_lbvh_sort_bench

A/B harness for the LBVH radix sort: checks every method against a host reference, then times single-block against multi-block.

| | |
|---|---|
| Group | `compute` |
| Sources | `src/compute/lbvh/bench/sort_bench.cpp`, `src/compute/lbvh/lbvh_sort.cpp`, `src/compute/lbvh/bench/bench_harness.cpp`, `src/compute/lbvh/bench/bench_stats.cpp` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_software_lbvh_sort_bench` |
| Run | `example_software_lbvh_sort_bench dx --headless` |

## Purpose

The radix sort is the second stage of the software LBVH build and, at scale, the
dominant one. This target isolates it: two implementations of the same stable
4 × 8-bit LSD sort are verified byte for byte and then timed, so a change can be
judged without the tree construction around it — and the crossover size is visible.

## Architecture

- **What is sorted.** `(Morton code, primitive slot)` pairs — four passes of eight
  bits, least significant digit first. Stability is part of the semantics: it makes
  four passes add up to a correct sort and two runs bit-identical, so both methods
  must produce the *same bytes*, not merely a sorted array.
- **The two methods (`lbvh_sort.cpp`).** `Method::single_block` is the reference
  implementation: one work-group walks the range in 256-element tiles and scatters with
  a bit-flag machine in shared memory, which costs ~3 barriers per tile with the rest
  of the device idle. `Method::multi_block` cuts the range into independent tiles of
  `block_size * items` elements and gives each one the same algorithm, which needs four
  dispatches per pass: per-block histogram → hierarchical `(block, digit)` exclusive
  scan → per-digit base → ranked stable scatter. `Method::automatic` keeps
  `single_block` below 8192 keys and uses `multi_block` above.
- **Correctness phase.** Before anything is timed, every method is run over a
  catalogue of element counts (0, 1, 2, 3, 31, 32, 33, 255 … 65536 and up), element
  widths (4, 8, 16 per chunk) and two base offsets — `--base 1237` sorts inside a
  buffer holding garbage before the range, as a real build does. Each case is
  compared element by element with a host sort and printed as `check …: ok`.
- **Timing phase.** For every count in `--counts` both methods run `--warmup` untimed
  and `--iters` timed samples; the harness reports min/median/mean milliseconds,
  Melem/s and the speedup against the single-block run at the same count.
- **Budgets and records.** The harness is shared with `example_software_lbvh_bench`:
  the memory estimate is checked against `--budget-gib` before anything is allocated,
  and a count whose predicted submission exceeds `--dispatch-budget-ms` is refused.
  Each timed configuration emits a `bench_sort` line (`method`, `count`, `base`,
  `ms_min`, `melem_per_s`) so several backends can be diffed.

## Usage

```bash
example_software_lbvh_sort_bench dx --headless
example_software_lbvh_sort_bench dx --check --base 1237
example_software_lbvh_sort_bench dx --measure --method multi --counts 262144,1048576
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` |
| `--check`, `--measure` | run only the correctness phase or only the timing phase; neither means both |
| `--method <all\|single\|multi\|auto>` | which implementation to time (default `all`) |
| `--counts <a,b,c>` | element counts of the timing sweep, default 4096 … 8388608 |
| `--check-counts <a,b,c>` | restrict the correctness catalogue (a debugging aid) |
| `--base <n>` | sort at a non-zero offset, i.e. with garbage before the range |
| `--items-sweep` | also sweep the elements per chunk of the parallel path |
| `--iters N`, `--warmup N`, `--frames N` | timed and untimed samples (`--frames` is a synonym for `--iters`) |
| `--budget-gib <f>`, `--dispatch-budget-ms <f>`, `--max-seconds <f>` | memory, submission and duration limits |
| `--headless`, `-h`, `--help` | no window (the harness never opens one); usage text |

## Result

Example console output on `dx`, abbreviated:

```text
sort_config backend=dx capacity=8388672 chunk=2048 scratch_bytes=8456192 warp_size=32 iters=5 warmup=1 seed=0
==== dx: correctness against the host reference ====
  check count=4096 base=1237 items=8: ok
==== dx: speedup against the single-block sort ====
  count method ms(min) Melem/s  vs single
  262144 multi0 0.3163 828.8 11.03x
  1048576 multi0 0.8237 1273.0 14.76x
dx: PASS
```

Below 8192 elements the single-block path wins (at 4096 the parallel path is ~0.91×),
which is exactly why `Method::automatic` switches there.

## See also

`example_software_lbvh` (where the sort is used), `example_software_lbvh_bench` (the
whole build and traversal), `example_software_lbvh_test` (the correctness suite).
