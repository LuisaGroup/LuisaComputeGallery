# example_coro_external_stage_debug

A coroutine suspends once per frame and a read-only external stage records the values carried across that suspension.

| | |
|---|---|
| Group | `coro` |
| Sources | `src/coro/external_stage_debug.cpp`, `src/common/headless.h` |
| Backends | dx, vk, cuda |
| Build | `xmake build example_coro_external_stage_debug` |
| Run | `example_coro_external_stage_debug dx --headless --frames 1` |

## Purpose

The smallest complete example of the coroutine external-stage protocol: 256
frames run a coroutine with one suspension, the wavefront scheduler parks them
there, a registered handler runs one device pass for the whole queue, and the
scheduler resumes them. The handler only observes, so the example also pins down
the strictest form of the contract: a stage that reads frame state, never writes.

## Architecture

- **Coroutine body.** `Coroutine<void(Buffer<uint>)>` computes `id = dispatch_x()`
  and `watched = id * 3 + 7`, suspends, then stores `watched * 2 + 1` into the
  output buffer. The suspension sits between computing and storing, so every value
  the coroutine still needs has to survive in the frame.
- **Suspension point.** `$suspend("after_transform", ...)` carries a
  `coro_annotation` for the schema `luisa.coro.debug.structured-watch` with
  `.fallback(CoroSuspendFallback::reject)` and two typed bindings,
  `.read("id", id)` and `.read("value", watched)`. `reject` means the program is
  not executed when no handler claims the stage, and the bindings are the only
  state an observer gets to see.
- **External stage.** `StructuredDebugger` declines any stage with another schema
  or version, asserts that `stage.dataflow->required_writeback_slot_span()` is
  empty, and compiles one `Kernel1D` bound to `stage.binding("id")` and
  `stage.binding("value")`. Per frame it rebuilds the parked frame
  (`CoroFrame::create` plus `coro_frame_load_into` with the stage's
  `reconstruct_slots`) and records `(id, value)` into a `Buffer<uint>` — no
  device-side printing, no whole-frame access.
- **Scheduler.** `WavefrontCoroScheduler<Buffer<uint>>` is built with
  `thread_count = 256`, `global_memory_soa`, `gather_by_sorting`,
  `frame_buffer_compaction`, `execution_block_size = 32`,
  `largest_continuation_first` and `report_stats`, so each dispatch prints the
  continuation queues it ran. The graph has two continuations (`<entry>`,
  `after_transform`) and one extension queue.
- **Stream and loop.** Device and one compute `Stream`
  (`device.create_stream(StreamTag::COMPUTE)`) are created once;
  `register_extension_handler(stream, ...)` prepares the stage on it and the
  handler appends its shader to `context.stream`, so the external pass finishes
  before the frames advance. `--frames N` repeats
  `stream << scheduler(output).dispatch(frame_count) << synchronize()`.
- **Checks.** Records and output are copied back and verified per frame
  (`records[2i] == i`, `records[2i + 1] == 3i + 7`, `output[i] == (3i + 7) * 2 + 1`),
  and `last_dispatch_stats()` must show one extension queue run once with 256 frames.

## Usage

```bash
example_coro_external_stage_debug dx --headless --frames 1  # the CI run
example_coro_external_stage_debug cuda --frames 8
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx`, `vk` or `cuda` |
| `--headless`, `--cli` | consumed for the shared convention; this example never opens a window, so the work is the same with or without it |
| `--frames N` | scheduler dispatches to run (default 1; `0` also means one) |
| `--output-dir DIR` | accepted and unused: the example writes no file |

The `coro` examples always use the wavefront scheduler; the `--scheduler` switch
belongs to the `rendering/` coro examples such as `example_coro_sdf_renderer`.

## Result

A console report of the coroutine shape and a pass/fail verdict:

```text
Wavefront continuation: index=0 token=0 name='<entry>' dispatches=1 executed=256 peak_queued=0.
Wavefront continuation: index=1 token=1 name='after_transform' dispatches=1 executed=256 peak_queued=256.
Wavefront Extension: queue=2 boundary=0 extension=0 schema='luisa.coro.debug.structured-watch' handler='structured-debugger' dispatches=1 executed=256 peak_queued=256.
Structured coroutine debug passed on 'vk': 256 frames through handler 'structured-debugger', with zero frame writeback slots (1 scheduler iteration(s), headless).
```

The exit code is 0 only when every record and every coroutine result matched; a
mismatch aborts the run with the offending frame index.

## See also

`example_coro_on_demand_texture` and `example_coro_external_stage_neural_sdf`
(stages that write frame state), `example_coro_sdf_renderer` (a coroutine renderer).
