# Headless Windows Nsight profiling

`nsight_profile.py` is a stdlib-only Python driver for NVIDIA Nsight Systems
(`nsys`) and Nsight Compute (`ncu`). It runs gallery examples with their existing
manifest-defined headless arguments, collects binary reports, and writes text
reports that can be read in a terminal. It never starts `nsys-ui`, `ncu-ui`, or
another GUI.

The requested Nsight Systems installation is detected automatically:

```text
C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1\target-windows-x64\nsys.exe
```

Other Nsight Systems/Compute versions under `C:\Program Files\NVIDIA Corporation`
are also detected, with the newest version preferred. Nsight Compute is optional
for `--mode nsys`, but is required for `--mode ncu` and `--mode both`.

## Quick start

Build at least one gallery target, check the installation, and preview the full
workflow without running anything:

```bat
xmake build example_ocr_bench
python scripts\nsight_profile.py doctor
python scripts\nsight_profile.py profile example_ocr_bench --mode both --launch-count 1 --dry-run
```

Collect a system trace and CUDA kernel analysis. The two profilers run separate
target invocations, sequentially:

```bat
python scripts\nsight_profile.py profile example_ocr_bench --backend cuda --mode both --launch-count 1 --run-name ocr-cuda
```

The text reports are ready when the command finishes:

```text
.kimix_cache\nsight\ocr-cuda\example_ocr_bench_cuda\system.stats.txt
.kimix_cache\nsight\ocr-cuda\example_ocr_bench_cuda\kernel.details.txt
```

No display is needed. The examples run with `--headless`; binary reports can be
copied to another machine later if GUI inspection is wanted.

## Commands

### `doctor`: installation and project diagnostics

```bat
python scripts\nsight_profile.py doctor
```

This prints the resolved `nsys`/`ncu` paths and versions, executable directory,
manifest target count, and platform status. A missing tool makes the command
return nonzero.

Tool lookup order is:

1. `--nsys PATH` / `--ncu PATH` (an executable or installation root).
2. `NSYS` / `NCU` environment variables.
3. `NSIGHT_SYSTEMS_ROOT` / `NSIGHT_COMPUTE_ROOT` environment variables.
4. `nsys` / `ncu` on `PATH`.
5. The requested Nsight Systems 2026.3.1 path and other installed versions.

For example:

```bat
python scripts\nsight_profile.py doctor --nsys "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1" --ncu "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1"
```

### `list`: inspect profileable targets

```bat
python scripts\nsight_profile.py list --filter rendering
python scripts\nsight_profile.py list --filter qwen --backend cuda
```

Only desktop binary targets from `gallery_targets.json` are listed. Static
libraries and iOS-only entries are excluded. Manifest `headless_skip` reasons
are shown rather than silently ignored.

### `profile`: collect reports

Select one or more exact target names, or use `--filter` for a name/group
substring. Targets are always profiled one at a time.

```bat
REM Nsight Systems: CUDA API, GPU kernels, memory operations, and NVTX
python scripts\nsight_profile.py profile example_path_tracing --backend cuda

REM Nsight Compute: detailed metrics for three matching CUDA kernel launches
python scripts\nsight_profile.py profile example_path_tracing --mode ncu --ncu-set detailed --launch-count 3

REM DX12 and Vulkan system traces (Nsight Compute is CUDA-only)
python scripts\nsight_profile.py profile example_path_tracing --backend dx --mode nsys
python scripts\nsight_profile.py profile example_path_tracing --backend vk --mode nsys

REM Multiple targets, selected by manifest group/name substring
python scripts\nsight_profile.py profile --filter rendering --backend cuda --mode nsys
```

The executable directory is auto-detected in this order:

```text
build\windows\x64\release
build\windows\x64\releasedbg
build\windows\x64\debug
build-cmake\bin
```

Pass `--exe-dir DIR` to override it. A missing binary is an error with a suggested
`xmake build <target>` command. `--dry-run` is the exception: it prints the planned
command even when tools or binaries are absent, and creates no files.

#### Target arguments

The driver constructs:

```text
<target.exe> <backend> --headless <manifest headless arguments> <extra arguments>
```

The backend spelling follows the manifest (`positional`, `flag`, or
`flag-equals`), and `$ROOT` in manifest arguments expands to the repository root.
Append extra target arguments with repeatable `--app-arg`. Use the equals form
when the value starts with a dash:

```bat
python scripts\nsight_profile.py profile example_path_tracing --no-manifest-args --app-arg=--spp --app-arg=4 --app-arg=--frames --app-arg=20
```

`--no-manifest-args` is important for targets whose default is only an inventory
operation such as `--list-versions`: replace that default with a real GPU
workload before expecting kernel metrics. `--include-skipped` overrides manifest
skip reasons, but missing model data or unsupported hardware can still make the
target fail.

#### Nsight Systems controls

Defaults are deliberately lightweight and do not require CPU sampling rights:

| Option | Default | Meaning |
|---|---|---|
| `--trace` | backend-specific | Comma-separated trace providers |
| `--duration` | `0` | Seconds to collect; zero means target exit |
| `--sample` | `none` | CPU instruction/backtrace sampling |
| `--cpuctxsw` | `none` | OS thread context-switch tracing |
| `--kill` | `true` | Terminate target when a collection session ends |
| `--resolve-symbols` | off | Resolve sampled/backtrace symbols on Windows |
| `--gpu-metrics` | off | Sample supported GPU metrics |
| `--reports` | backend-specific | Reports passed to `nsys stats` |
| `--nsys-arg` | none | One advanced profile argument; repeatable |

Automatic trace/report sets are:

| Backend | Trace providers | Default text summaries |
|---|---|---|
| `cuda` | `cuda,nvtx` | CUDA API, GPU kernels, memory time/size, NVTX |
| `dx` | `dx12,dx12-annotations,nvtx` | DX12 PIX ranges, DX12 GPU markers, NVTX |
| `vk` | `vulkan,vulkan-annotations,nvtx` | Vulkan API, CPU/GPU markers, NVTX |

A finite capture can be requested with `--duration 20`. With the default
`--kill true`, the target is terminated when that session ends. Use
`--kill false` only when leaving the target running is explicitly desired.

For lower-level Windows queue data, add WDDM explicitly (additional privileges
may be required):

```bat
python scripts\nsight_profile.py profile example_path_tracing --backend dx --trace=dx12,dx12-annotations,nvtx,wddm --reports=dx12_pix_sum,dx12_gpu_marker_sum,wddm_queue_sum,nvtx_sum
```

CPU sampling, context-switch tracing, some GPU metrics, and extended WDDM events
can require an elevated terminal and are system-dependent. Collection errors are
preserved in the logs and produce a nonzero script exit code.

#### Nsight Compute controls (CUDA only)

| Option | Default | Meaning |
|---|---|---|
| `--ncu-set` | `basic` | Section set (`basic`, `detailed`, `full`, `roofline`, ...) |
| `--launch-count` | `10` | Maximum matching kernel launches |
| `--all-launches` | off | Omit the launch limit; potentially very slow |
| `--launch-skip` | `0` | Matching launches to skip |
| `--kernel-name` | none | Exact kernel name or `regex:<expression>` |
| `--clock-control` | `base` | Lock clocks to base for stable profiling |
| `--cache-control` | `all` | Control/reset GPU caches during replay |
| `--page` | `details` | Text page (`details` or `raw`) |
| `--csv` | off | Write Nsight Compute text as CSV |
| `--ncu-arg` | none | One advanced profile argument; repeatable |

Example:

```bat
python scripts\nsight_profile.py profile example_path_tracing --mode ncu --ncu-set full --kernel-name regex:.*render.* --launch-count 1 --page raw
```

Nsight Compute may replay kernels, lock clocks, and flush caches. Its timings and
target runtime therefore differ from an unprofiled run. Start with `basic` and a
small launch count; `full` plus `--all-launches` can be very expensive. Use
`--clock-control none` or `--cache-control none` only when that trade-off is
intentional.

### `report`: regenerate text without rerunning the target

Pass binary reports, a run directory, or a mixture. Directories are searched
recursively for `.nsys-rep` and `.ncu-rep` files.

```bat
python scripts\nsight_profile.py report .kimix_cache\nsight\ocr-cuda --backend cuda
python scripts\nsight_profile.py report path\to\system.nsys-rep --backend dx
python scripts\nsight_profile.py report path\to\kernel.ncu-rep --page raw
python scripts\nsight_profile.py report path\to\kernel.ncu-rep --csv --output-dir path\to\text
```

The default output is beside each binary report. Existing text files are not
overwritten unless `--force` is given. `nsys stats` also exports/refreshes the
associated SQLite database. Use `--reports` to change the Nsight Systems report
set and `--backend` to select the appropriate built-in defaults.

## Output layout and failures

Each profile invocation creates a timestamped run directory, or the directory
named by `--run-name`:

```text
.kimix_cache\nsight\<run>\ 
  summary.json
  <target>_<backend>\ 
    nsys.profile.log
    system.nsys-rep
    system.sqlite
    system.stats.txt
    ncu.profile.log
    kernel.ncu-rep
    kernel.details.txt
```

Only files relevant to the selected mode are produced. `--no-report` skips text
generation while retaining binary reports and profiler logs. `summary.json`
contains every argv array, printable command, log/report path, return code,
status, and elapsed time. Output from the target and profiler is redirected to
these logs; `nsys` is explicitly invoked with `--show-output=true`.

An existing run directory is refused unless `--force` is supplied. A profiler
failure, timeout, missing expected report, text-report failure, or missing binary
makes the script return nonzero. Because `nsys` propagates the target's exit code,
a failed target can still leave a useful `.nsys-rep`; when that file exists, text
statistics are still generated while the overall run remains failed. In `--mode
both`, the second profiler is still attempted if the first fails because their
collections are independent.

Use `--timeout SECONDS` to bound each profiler/report process (default: 900).
As with other command-line profilers, killing a timed-out parent cannot guarantee
cleanup of every externally launched child; check the Nsight session/target state
after an interrupted collection.

## Tests

The unit tests do not require an NVIDIA GPU or run the profilers:

```bat
python -m unittest discover -s scripts -p test_nsight_profile.py -v
```

They cover command construction, backend-specific traces, manifest expansion,
tool discovery, overwrite/dry-run behavior, report collection, logs, and failure
propagation. Use `doctor` and a real `--dry-run` before a GPU collection.

## Official CLI references

- Nsight Systems User Guide, Profiling from the CLI:
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli
- Nsight Compute CLI:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Installed help: `nsys --help`, `nsys profile --help`, `nsys stats --help`,
  `nsys stats --help-reports`, and `ncu --help`.
