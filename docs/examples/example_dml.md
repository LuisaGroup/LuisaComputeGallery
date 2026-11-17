# example_dml

Builds a two-layer neural network as a DirectML graph, runs one forward pass on it, and prints the values DirectML produced.

| | |
|---|---|
| Group | `extension` |
| Sources | `src/extension/dml.cpp`, `src/common/headless.h` |
| Backends | dx |
| Build | `xmake build example_dml` |
| Run | `example_dml dx --headless` |

## Purpose

DirectML is the operator-level machine-learning API of D3D12: you describe a
network, DirectML fuses and compiles it, and one command executes it. This example
is the smallest honest use of that — a 2 → 2 → 2 network, two fused activations,
one `forward` call — and it leaves the numbers visible so the result can be
checked by hand.

## Architecture

- **Device.** `ctx.create_device("dx")`, unconditionally: `DirectMLExt` only exists
  on the DirectX backend. The positional command-line word labels the run, it does
  not select a device.
- **Graph description.** `device.extension<DirectMLExt>()`, then
  `ext->create_graph(batch_size, input_elements, out_elements,
  hidden_layer_elements, activations, half_precision)`. The example passes
  `create_graph(1u, 2u, 2u, hidden_layer /* {2} */, activation, false)`: batch 1,
  two inputs, two outputs, one hidden layer of two elements, `half_precision` off.
- **Activations.** `FusedActivation activation[]` carries one entry per layer —
  here `FusedActivation::none()` twice, so the graph is a plain affine chain. The
  commented `FusedActivation::scaled_elu()` / `sigmoid()` turn it into a non-linear
  network; DirectML folds the activation into the preceding operator.
- **Build.** `stream << graph->build()` submits the compiled operator before any
  buffer is touched, so the graph is final when `forward` is encoded.
- **Buffers sized from the graph.** `weight_buffer_size_bytes()` and
  `output_buffer_size_bytes()` are asked instead of being derived by hand; the host
  vectors are resized in floats from those byte counts. The weights are the eight
  values `5, 6, 7, 8, 11, 12, 13, 14`, the input is `{3, 4}`.
- **Dispatch and readback.** One stream carries data, weights, computation and
  result: `w.copy_from(...) << ipt.copy_from(...) << graph->forward(ipt, opt, w) <<
  opt.copy_to(...) << synchronize()`. `forward` takes input, output and weight
  buffer and nothing else — no per-layer plumbing reaches the caller.
- **Expected numbers.** The source spells out the two matrix steps next to the
  weights: `[[5, 6], [7, 8]] · [3, 4] = [39, 53]`, then
  `[[11, 12], [13, 14]] · [39, 53] = [1065, 1249]`. The file also carries
  `NNLayerRelu`, `NNLayer` and `NNForward`, a plain C++ rendering of the same
  arithmetic, which is how those numbers were derived.

## Usage

```bash
example_dml dx # no window; the whole demo is the console report
example_dml dx --headless
```

| Option | Meaning |
|---|---|
| `<backend>` | accepted and ignored: the device is always the `dx` backend |
| `--headless`, `--cli` | no window; accepted for the uniform gallery CLI — the example never opens one |
| `--frames N`, `--output-dir DIR` | consumed by the shared parser and unused: the work is one forward pass |

## Result

Example console output on `dx`:

```text
[info] Output: 0 is: 1065
[info] Output: 1 is: 1249
[info] dml: success (2 output value(s), headless=true).
```

`1065` and `1249` are exactly the hand-computed values above, so a passing run
means the graph got the intended shape, the weight buffer was laid out the way
DirectML expected, and the result came back through `opt`. Swap in the commented
`FusedActivation::sigmoid()` or `scaled_elu()` and the same two lines report the
activated result: the first layer's partial sums `39` and `53` are squashed before
the second matrix is applied.

## See also

`example_dstorage` (DirectStorage instead of DirectML), `tokenizer` and
`example_helloworld` (kernels you write yourself rather than an operator library).
