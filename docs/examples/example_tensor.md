# example_tensor

The Tile DSL surface of LuisaCompute: 26 kernels captured as TileIR and verified against host references, plus seven training and inference demos.

| | |
|---|---|
| Group | `tensor` |
| Sources | `src/tensor/main.cpp`, `src/tensor/tensor_kernels.cpp`, `src/tensor/kernel_*.cpp`, `src/tensor/tensor_basics.cpp`, `src/tensor/poly_fit.cpp`, `src/tensor/linear_regression.cpp`, `src/tensor/mlp.cpp`, `src/tensor/mnist.cpp`, `src/tensor/rnn.cpp`, `src/tensor/cnn_inference.cpp` |
| Backends | dx, vk |
| Build | `xmake build example_tensor` |
| Run | `example_tensor dx --headless --kernel rms_norm` |

## Purpose

A tour of the tensor/Tile DSL: every kernel is captured with
`tile::tile_kernel(...).capture()`, verified, compiled for the backend, dispatched
on real buffers and compared with a host reference. Use it to see which tile
operations a backend lowers correctly.

## Architecture

- **Selection (`main.cpp`).** Shared flags are consumed first, then `--cnn`,
  `--poly-fit`, `--linear-regression`, `--mlp`, `--mnist`, `--rnn` and `--basics`
  hand over to their own driver; without one the kernel suite runs its 26
  `KernelCase` entries, narrowed by a repeatable `--kernel <name>`.
- **Registry.** Elementwise maths, `pipelined_matmul`, `rms_norm`, `softmax`,
  `tile_fill`/`tile_transpose`/`tile_clamp`, the tile collectives (`tile_reduce`,
  `tile_scan`, `tile_min_abs`, `tile_sync`, `tile_warp_reduce`), the unary maths,
  `cast`, `pow` and `dtypes`; several report more than one check, so a full run
  ends at 37.
- **Compile path (`compile_tile` in `tensor_kernels.cpp`).** capture →
  `tile::verify` → `tile::compile` with `Lowering::NATIVE`; a backend that answers
  with a TIRX diagnostic is retried with `Lowering::TIRX`. Each compile logs its
  dispatch shape and realization chain, e.g. `TileIR -> XIR SSA -> AST -> SPIR-V`.
- **Accounting.** `record`, `check` and `skip` print
  `[tensor] PASS name — max err = ... (tol ...)`, and the exit code is the number of
  failed checks. `--guard` checks both halves of the two-root-parallel contract
  (capture and verify accept the structure, lowering rejects it); `--report` lists
  what the new TileIR cannot express: atomics, warp vote and shuffle, loop break,
  sin/cos, fp8/i4/fp4, device print, explicit shared memory.
- **Demos.** `--basics` walks four exercises (shapes, an add/mul kernel, autograd
  for `y = x² + 2x + 1` at `x = 3`, a 1→1 net trained 2000 steps);
  `--poly-fit [--steps N]` fits a degree-3 polynomial to `sin(x)` (2000 steps)
  against a host reference; `--linear-regression [--steps N]` trains linear and
  logistic models (200); `--mlp [--epochs N]` a 50→30→15→4 net (60);
  `--mnist [--epochs N]` a 64→32→10 net on synthetic TinyMNIST (30);
  `--rnn [--epochs N]` a tanh RNN on counting sequences (150); `--cnn` TinyCNN
  inference. Thresholds match those budgets, so a smaller budget fails.
- **Backends.** `dx` and `vk` run the native lowering; CUDA needs the optional TIRx
  bridge (`LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE`), off in this build.

## Usage

```bash
example_tensor dx --headless --kernel rms_norm   # one registry case
example_tensor dx                                # all 26 cases, 37 checks
example_tensor vk --headless --poly-fit --steps 500
example_tensor dx --cnn src/tensor/cnn_input.bin
```

| Option | Meaning |
|---|---|
| `<backend>` | `dx` or `vk` in this build; CUDA needs the TIRx bridge |
| `--kernel <name>` | run one registry case; repeat the flag for several, omit it for all 26 |
| `--guard`, `--report` | two-root-parallel contract test; unported-feature list |
| `--basics`, `--poly-fit`, `--linear-regression`, `--mlp`, `--mnist`, `--rnn` | the demos described above |
| `--cnn [file.bin]`, `--bench` | TinyCNN inference plus a timing run; the model defaults to `cnn_input.bin` in the working directory, so pass `src/tensor/cnn_input.bin` from the repository root |
| `--steps N`, `--epochs N` | budget of `--poly-fit`/`--linear-regression` and of `--mlp`/`--mnist`/`--rnn` |
| `--headless`, `--cli`, `--frames N`, `--output-dir DIR` | the shared convention flags; no window is opened, no image is written, `--frames` is unused |

## Result

```text
=== tensor: running 26 kernel case(s) on 'vk' ===
--- tensor: case 'rms_norm' ---
[tensor] PASS  rms_norm —
[tensor] max err = 3.22333e-07 (tol 0.001)
...
=== tensor: 37 passed, 0 failed (headless=true) ===
```

The check count follows the filters: `--kernel rms_norm` ends at `1 passed`, adding
`--kernel tile_scan` makes it `3 passed`. A failing check prints the backend
diagnostic, so the exit code alone is enough for a script; demos end with e.g.
`[poly-fit] Verification: PASSED`.

## See also

`example_multi_head_attention` (attention kernels against a CPU reference),
`example_helloworld` (the plain kernel dispatch).
