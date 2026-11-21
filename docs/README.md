# LuisaComputeGallery examples

Every example in this repository is a self-contained program that shows one part
of [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute): the DSL, the
runtime, ray tracing, the coroutine module, tensor programming, and the higher
level engines (embedding, OCR, neural texture compression).

Each page below documents one runnable target: what it is for, how it is
architected, how to run it, and - when it draws something - the picture it
produces. Every picture in these pages was captured with the built-in capture
mode (`--capture result.jpg`), so any of them can be reproduced with one command.

- Build: [build.md](build.md) (`xmake build <target>`, or the CMake mirror).
- Run without a window: [headless.md](headless.md) (`<example> <backend> --headless`).
- Save the result picture: [capture.md](capture.md) (`--capture <file.png|jpg>`).
- Example data (weights, textures): `python scripts/unpack_data.py`;
  `python scripts/pack_data.py` publishes a new package.

The `Backends` column is copied from each page and lists the backends the target
really uses: `dx` (DirectX 12), `vk` (Vulkan) and `cuda` (NVIDIA). A page narrows
it when the target is not portable - an example that always creates a `dx` device,
or one whose feature only exists on some backends, says so in the same cell.

## Gallery

| <a href="examples/example_path_tracing.md"><img src="images/example_path_tracing.jpg" width="220"></a> | <a href="examples/example_path_tracing_camera.md"><img src="images/example_path_tracing_camera.jpg" width="220"></a> | <a href="examples/example_path_tracing_cutout.md"><img src="images/example_path_tracing_cutout.jpg" width="220"></a> | <a href="examples/example_path_tracing_hdr.md"><img src="images/example_path_tracing_hdr.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_path_tracing](examples/example_path_tracing.md) | [example_path_tracing_camera](examples/example_path_tracing_camera.md) | [example_path_tracing_cutout](examples/example_path_tracing_cutout.md) | [example_path_tracing_hdr](examples/example_path_tracing_hdr.md) |
| <a href="examples/example_path_tracing_nested_callable.md"><img src="images/example_path_tracing_nested_callable.jpg" width="220"></a> | <a href="examples/example_path_tracing_ray_masks.md"><img src="images/example_path_tracing_ray_masks.jpg" width="220"></a> | <a href="examples/example_path_tracing_spectrum.md"><img src="images/example_path_tracing_spectrum.jpg" width="220"></a> | <a href="examples/example_photon_mapping.md"><img src="images/example_photon_mapping.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_path_tracing_nested_callable](examples/example_path_tracing_nested_callable.md) | [example_path_tracing_ray_masks](examples/example_path_tracing_ray_masks.md) | [example_path_tracing_spectrum](examples/example_path_tracing_spectrum.md) | [example_photon_mapping](examples/example_photon_mapping.md) |
| <a href="examples/example_sdf_renderer.md"><img src="images/example_sdf_renderer.jpg" width="220"></a> | <a href="examples/example_blackhole.md"><img src="images/example_blackhole.jpg" width="220"></a> | <a href="examples/example_voxel_raytracer.md"><img src="images/example_voxel_raytracer.jpg" width="220"></a> | <a href="examples/example_procedural.md"><img src="images/example_procedural.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_sdf_renderer](examples/example_sdf_renderer.md) | [example_blackhole](examples/example_blackhole.md) | [example_voxel_raytracer](examples/example_voxel_raytracer.md) | [example_procedural](examples/example_procedural.md) |
| <a href="examples/example_shader_toy.md"><img src="images/example_shader_toy.jpg" width="220"></a> | <a href="examples/example_shader_toy_spacex.md"><img src="images/example_shader_toy_spacex.jpg" width="220"></a> | <a href="examples/example_shader_visuals_present.md"><img src="images/example_shader_visuals_present.jpg" width="220"></a> | <a href="examples/example_path_tracing_xir2ast.md"><img src="images/example_path_tracing_xir2ast.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_shader_toy](examples/example_shader_toy.md) | [example_shader_toy_spacex](examples/example_shader_toy_spacex.md) | [example_shader_visuals_present](examples/example_shader_visuals_present.md) | [example_path_tracing_xir2ast](examples/example_path_tracing_xir2ast.md) |
| <a href="examples/example_sdf_renderer_xir2ast.md"><img src="images/example_sdf_renderer_xir2ast.jpg" width="220"></a> | <a href="examples/example_coro_sdf_renderer.md"><img src="images/example_coro_sdf_renderer.jpg" width="220"></a> | <a href="examples/example_coro_path_tracing.md"><img src="images/example_coro_path_tracing.jpg" width="220"></a> | <a href="examples/example_coro_path_tracing_wavefront.md"><img src="images/example_coro_path_tracing_wavefront.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_sdf_renderer_xir2ast](examples/example_sdf_renderer_xir2ast.md) | [example_coro_sdf_renderer](examples/example_coro_sdf_renderer.md) | [example_coro_path_tracing](examples/example_coro_path_tracing.md) | [example_coro_path_tracing_wavefront](examples/example_coro_path_tracing_wavefront.md) |
| <a href="examples/example_fire_simulation.md"><img src="images/example_fire_simulation.jpg" width="220"></a> | <a href="examples/example_game_of_life.md"><img src="images/example_game_of_life.jpg" width="220"></a> | <a href="examples/example_mpm3d.md"><img src="images/example_mpm3d.jpg" width="220"></a> | <a href="examples/example_mpm88.md"><img src="images/example_mpm88.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_fire_simulation](examples/example_fire_simulation.md) | [example_game_of_life](examples/example_game_of_life.md) | [example_mpm3d](examples/example_mpm3d.md) | [example_mpm88](examples/example_mpm88.md) |
| <a href="examples/example_nbody_simulation.md"><img src="images/example_nbody_simulation.jpg" width="220"></a> | <a href="examples/example_wave_equation.md"><img src="images/example_wave_equation.jpg" width="220"></a> | <a href="examples/example_imgui.md"><img src="images/example_imgui.jpg" width="220"></a> | <a href="examples/example_swapchain.md"><img src="images/example_swapchain.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_nbody_simulation](examples/example_nbody_simulation.md) | [example_wave_equation](examples/example_wave_equation.md) | [example_imgui](examples/example_imgui.md) | [example_swapchain](examples/example_swapchain.md) |
| <a href="examples/example_swapchain_static.md"><img src="images/example_swapchain_static.jpg" width="220"></a> | <a href="examples/example_win_hdr.md"><img src="images/example_win_hdr.jpg" width="220"></a> | <a href="examples/example_image_process.md"><img src="images/example_image_process.jpg" width="220"></a> | <a href="examples/example_procedural_tex.md"><img src="images/example_procedural_tex.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_swapchain_static](examples/example_swapchain_static.md) | [example_win_hdr](examples/example_win_hdr.md) | [example_image_process](examples/example_image_process.md) | [example_procedural_tex](examples/example_procedural_tex.md) |
| <a href="examples/example_image_processing.md"><img src="images/example_image_processing.jpg" width="220"></a> | <a href="examples/example_coro_external_stage_neural_sdf.md"><img src="images/example_coro_external_stage_neural_sdf.jpg" width="220"></a> | <a href="examples/example_coro_on_demand_texture.md"><img src="images/example_coro_on_demand_texture.jpg" width="220"></a> | <a href="examples/luisa_ntc_viewer.md"><img src="images/luisa_ntc_viewer.jpg" width="220"></a> |
| :-: | :-: | :-: | :-: |
| [example_image_processing](examples/example_image_processing.md) | [example_coro_external_stage_neural_sdf](examples/example_coro_external_stage_neural_sdf.md) | [example_coro_on_demand_texture](examples/example_coro_on_demand_texture.md) | [luisa_ntc_viewer](examples/luisa_ntc_viewer.md) |


## rendering

| Example | What it shows | Backends |
|---|---|---|
| [example_path_tracing](examples/example_path_tracing.md) | Unidirectional path tracer for the Cornell Box, with next event estimation, MIS and Russian roulette. | dx, vk, cuda |
| [example_path_tracing_camera](examples/example_path_tracing_camera.md) | The Cornell Box path tracer driven by an interactive first-person camera you fly through the scene. | dx, vk, cuda |
| [example_path_tracing_cutout](examples/example_path_tracing_cutout.md) | The Cornell Box with striped alpha cut-outs in both boxes, traced through five ray-query strategies. | dx, vk, cuda |
| [example_path_tracing_hdr](examples/example_path_tracing_hdr.md) | The Cornell Box path tracer presenting into an HDR swapchain, with sRGB, PQ (ST.2084) and linear output paths. | dx, vk, cuda |
| [example_path_tracing_nested_callable](examples/example_path_tracing_nested_callable.md) | The Cornell Box path tracer written with nested DSL lambdas, so NEE and BSDF sampling become reusable blocks. | dx, vk, cuda |
| [example_path_tracing_ray_masks](examples/example_path_tracing_ray_masks.md) | The Cornell Box path tracer using instance visibility masks to hide one box from camera rays and the other from shadow rays. | dx, vk, cuda |
| [example_path_tracing_spectrum](examples/example_path_tracing_spectrum.md) | Spectral path tracing next to RGB path tracing in one image, with reflectances stored as Fourier moments. | dx, vk |
| [example_photon_mapping](examples/example_photon_mapping.md) | Two-pass photon mapping for the Cornell Box: photons are traced into a uniform grid, then gathered by eye rays. | dx, vk, cuda |
| [example_sdf_renderer](examples/example_sdf_renderer.md) | A ray-marching path tracer over an analytic signed distance field — no meshes, no acceleration structure. | dx, vk, cuda |
| [example_blackhole](examples/example_blackhole.md) | A Schwarzschild black hole with gravitational lensing, a Doppler-beamed accretion disk and a procedural starfield. | dx, vk, cuda |
| [example_voxel_raytracer](examples/example_voxel_raytracer.md) | Voxel ray tracer: walks a 64×64×64 voxel grid with a DDA loop and shades the first solid cell it meets. | dx, vk, cuda |
| [example_procedural](examples/example_procedural.md) | Ray-query renderer for procedural primitives: 1024 spheres that exist only as AABBs, plus a triangle mesh whose hits are filtered by a surface-candidate hook. | dx, vk, cuda |
| [example_shader_toy](examples/example_shader_toy.md) | ShaderToy-style ray marcher: an animated scene defined by a signed distance function, shaded by distance-based colour accumulation instead of a light. | dx, vk, cuda |
| [example_shader_toy_spacex](examples/example_shader_toy_spacex.md) | XorDev's "Starship" shader ported to LuisaCompute: one full-screen kernel that builds a fiery particle field by sampling a bindless texture fifty times. | dx, vk, cuda |
| [example_shader_visuals_present](examples/example_shader_visuals_present.md) | A beat-synchronised SDF showreel shader: an 80-step march with reflections, glow accumulation, fog and a material palette driven by the same distance field. | dx, vk, cuda |
| [example_path_tracing_xir2ast](examples/example_path_tracing_xir2ast.md) | The Cornell Box path tracer written in the LuisaCompute DSL, recorded to AST, translated to XIR and lowered back to AST before it is compiled for the backend. | dx, vk, cuda |
| [example_sdf_renderer_xir2ast](examples/example_sdf_renderer_xir2ast.md) | The Taichi-style SDF path tracer, recorded to AST, translated to XIR and lowered back to AST before it is compiled for the backend. | dx, vk, cuda |
| [example_coro_sdf_renderer](examples/example_coro_sdf_renderer.md) | The SDF path tracer written as a LuisaCompute coroutine: the per-pixel path is split into resumable subroutines and executed by a selectable scheduler. | dx, vk, cuda |
| [example_coro_path_tracing](examples/example_coro_path_tracing.md) | Cornell Box path tracing written as a LuisaCompute coroutine, with a switchable dispatch shape and a selectable scheduler. | dx, vk, cuda |
| [example_coro_path_tracing_wavefront](examples/example_coro_path_tracing_wavefront.md) | The coroutine path tracer built with the wavefront dispatch shape as its default: one coroutine per sample, launched as a 3-D dispatch. | dx, vk, cuda |

## simulation

| Example | What it shows | Backends |
|---|---|---|
| [example_fire_simulation](examples/example_fire_simulation.md) | A 65 536-particle fire and smoke system: particles are integrated on the GPU, splatted with a temperature colour ramp, and verified particle by particle in the offline path. | dx, vk, cuda |
| [example_game_of_life](examples/example_game_of_life.md) | Conway's Game of Life on a 128x128 toroidal grid, one cell per texel, run with a ping-pong image pair. | dx, vk, cuda |
| [example_mpm3d](examples/example_mpm3d.md) | A 3D Material Point Method solver: 65 536 particles scatter to a 64^3 grid and gather back every substep, then get drawn with an isometric projection. | dx, vk, cuda |
| [example_mpm88](examples/example_mpm88.md) | The 2D Material Point Method: 20 000 particles transfer momentum to a 200x200 grid and back, 24 substeps per frame. | dx, vk, cuda |
| [example_nbody_simulation](examples/example_nbody_simulation.md) | A 2048-body gravitational simulation with an all-pairs force kernel and a depth-resolved, deterministic particle rasteriser. | dx, vk, cuda |
| [example_wave_equation](examples/example_wave_equation.md) | A finite-difference solver for the 2D wave equation on a 512x512 grid, shaded as rippling water with caustics. | dx, vk, cuda |

## gui

| Example | What it shows | Backends |
|---|---|---|
| [example_imgui](examples/example_imgui.md) | Dear ImGui hosted on a LuisaCompute swapchain: a compute shader paints the background and ImGui draws its widgets on top. | dx, vk, cuda |
| [example_swapchain](examples/example_swapchain.md) | One device, four windows, four swapchains: the same rendered image presented to every window each frame. | dx, vk, cuda |
| [example_swapchain_static](examples/example_swapchain_static.md) | Loads a PNG, uploads it once and presents the same still image through a swapchain. | dx, vk, cuda |
| [example_win_hdr](examples/example_win_hdr.md) | HDR presentation through the DirectX HDR extension: query the display, pick a colour space, and encode the frame for it in a compute shader. | dx, vk, cuda (headless); dx for the HDR window |
| [example_image_process](examples/example_image_process.md) | An image-processing pipeline driven from an ImGui window: an ordered list of per-pixel operators applied in one compute dispatch, plus a full headless self test. | dx, vk, cuda |
| [example_procedural_tex](examples/example_procedural_tex.md) | Sixteen procedural-texture algorithms, each with its own kernel, parameter panel and PNG output. | dx, vk, cuda |

## compute

| Example | What it shows | Backends |
|---|---|---|
| [example_helloworld](examples/example_helloworld.md) | The smallest complete LuisaCompute program: one kernel writes a struct into a buffer and the host reads it back. | dx, vk, cuda |
| [example_bindless_mip](examples/example_bindless_mip.md) | Reads and samples every mip level of one image through a bindless heap, and shows that both routes resolve the same texel. | dx, vk, cuda |
| [example_async_copy_prefetch](examples/example_async_copy_prefetch.md) | Two-stage `cp.async` prefetch into shared memory, using the pipeline primitives instead of a barrier per tile. | cuda |
| [example_software_lbvh](examples/example_software_lbvh.md) | A two-level BVH built entirely by hand in Luisa kernels — no hardware RTX API — validated ray by ray against Luisa's own RTX traversal. | dx, vk, cuda |
| [example_software_lbvh_bench](examples/example_software_lbvh_bench.md) | Performance benchmark for the software LBVH: adversarial scenes through the build and the traversal, with a per-stage breakdown and traversal counters. | dx, vk, cuda |
| [example_software_lbvh_sort_bench](examples/example_software_lbvh_sort_bench.md) | A/B harness for the LBVH radix sort: checks every method against a host reference, then times single-block against multi-block. | dx, vk, cuda |
| [example_software_lbvh_test](examples/example_software_lbvh_test.md) | The correctness suite of the software LBVH: boundary sizes, degenerate geometry, determinism and compaction, each compared with a host reference and with the RTX structure. | dx, vk, cuda |
| [example_image_processing](examples/example_image_processing.md) | A five-kernel image pipeline — procedural pattern, separable Gaussian blur, Sobel edges, composite — with a window and a headless image writer. | dx, vk, cuda |
| [example_native_shader](examples/example_native_shader.md) | Compiles hand-written HLSL, GLSL or CUDA C++ through `NativeShaderExt` and replays a JSON dispatch document that declares the shaders, the resources and the workflow. | dx, vk, cuda |
| [example_native_shader_minimal](examples/example_native_shader_minimal.md) | The smallest complete native-shader example: one file that compiles a hand-written kernel (or a vertex/pixel pair), dispatches it and verifies the result on the host. | dx, vk, cuda |
| [example_gdeflate](examples/example_gdeflate.md) | A GDeflate (DirectStorage) compressor and decompressor written as two Luisa kernels, exercised by a round trip, hand-built Huffman tiles and malformed-stream corner cases. | dx, vk, cuda |
| [tokenizer](examples/tokenizer.md) | N-gram speculative drafting on the device: a corpus is tokenized on the host, uploaded as a flat token-ID library, and a DSL kernel finds the earliest match of the query's trailing n-gram and proposes the k tokens... | dx, vk, cuda |

## tensor

| Example | What it shows | Backends |
|---|---|---|
| [example_tensor](examples/example_tensor.md) | The Tile DSL surface of LuisaCompute: 26 kernels captured as TileIR and verified against host references, plus seven training and inference demos. | dx, vk |

## ml

| Example | What it shows | Backends |
|---|---|---|
| [example_multi_head_attention](examples/example_multi_head_attention.md) | Four attention paths — fused MHA, MLA, cooperative-vector MLA and paged MHA — on one fixed problem, each verified against a CPU reference. | dx, vk, cuda |

## coro

| Example | What it shows | Backends |
|---|---|---|
| [example_coro_external_stage_debug](examples/example_coro_external_stage_debug.md) | A coroutine suspends once per frame and a read-only external stage records the values carried across that suspension. | dx, vk, cuda |
| [example_coro_external_stage_neural_sdf](examples/example_coro_external_stage_neural_sdf.md) | Path tracer whose distance and normal queries are answered by external stages, keeping a neural bunny SDF outside the coroutine. | dx, vk, cuda |
| [example_coro_on_demand_texture](examples/example_coro_on_demand_texture.md) | A virtual texture sampled from coroutines: each miss suspends the pixel, a host-side cache loads one page, and the scheduler resumes the waiters. | dx, vk, cuda |

## extension

| Example | What it shows | Backends |
|---|---|---|
| [example_dstorage](examples/example_dstorage.md) | Streams a scratch file and a pinned host range into device buffers and into a 2D image through the DirectStorage extension, then compares every byte. | dx, vk, cuda |
| [example_dml](examples/example_dml.md) | Builds a two-layer neural network as a DirectML graph, runs one forward pass on it, and prints the values DirectML produced. | dx |

## dx

| Example | What it shows | Backends |
|---|---|---|
| [example_dx_custom_sdk](examples/example_dx_custom_sdk.md) | Creates the DirectX device through a `DirectXDeviceConfigExt` subclass that asks for its own Agility SDK version and DLL directory. | dx |

## interop

| Example | What it shows | Backends |
|---|---|---|
| [example_cuda_dx_interop](examples/example_cuda_dx_interop.md) | Shares one buffer between a DirectX and a CUDA device, orders the two transfers with a shared timeline semaphore, and reads the value back. | dx, cuda |
| [example_cuda_vk_interop](examples/example_cuda_vk_interop.md) | Shares one buffer between a Vulkan and a CUDA device, with the CUDA backend created on the very `VkDevice` the Vulkan device owns. | vk, cuda |
| [example_cuda_vk_kernel_launch](examples/example_cuda_vk_kernel_launch.md) | Compiles DSL kernels with the CUDA backend and dispatches them inside a Vulkan command buffer through `VK_NV_cuda_kernel_launch`. | vk, cuda |

## embedding

| Example | What it shows | Backends |
|---|---|---|
| [example_embed_retrieval](examples/example_embed_retrieval.md) | Whole-file hybrid retrieval over a directory: every document and every query is encoded by the bge-m3 session, and the index fuses the dense cosine with the sparse lexical cosine (0.7 / 0.3). | dx, vk, cuda |
| [example_embed_dump_golden](examples/example_embed_dump_golden.md) | Dumps the per-layer activations of one encoded sentence to raw `fp32` files, optionally comparing them against pinned golden layer states. | dx, vk, cuda |
| [example_embed_text](examples/example_embed_text.md) | Encodes sentences with the bge-m3 embedding engine on the GPU and prints the dense vector plus, on request, the sparse lexical and ColBERT heads. | dx, vk, cuda |

## embedding_mm

| Example | What it shows | Backends |
|---|---|---|
| [example_embed_mm_test](examples/example_embed_mm_test.md) | Self-check and PyTorch-parity suite for the EmbeddingGemma-2 multimodal engine: tokenizer, weight package, feature bundles, every GPU operator, and layer-by-layer drift against the official tutorial outputs. | dx, vk, cuda |

## ocr

| Example | What it shows | Backends |
|---|---|---|
| [example_ocr](examples/example_ocr.md) | Command-line front end of the LuisaCompute OCR engine: validates a complete recognition request, prints the model support report offline, and refuses unvalidated models instead of inventing text. | none (offline tool, no device) |
| [example_ocr_bench](examples/example_ocr_bench.md) | Correctness-gated benchmark harness for the OCR engine: runs a suite of synthetic workloads on the device and reports latency statistics, counters and correctness flags as JSON. | dx, vk, cuda |

## qwen

| Example | What it shows | Backends |
|---|---|---|
| [example_qwen](examples/example_qwen.md) | Text-only, single-sequence Qwen3.5-4B inference on LuisaCompute: six explicitly selectable engine variants (ordinary DSL or Tile kernels × BF16, true FP8 E4M3, native Q8_0 int8-block or W4A16 group-wise weights), an... | dsl: cuda, dx, vk; tile: dx, vk (real-model inference and device suite); Tile CUDA is unavailable in this build |

## ntc

| Example | What it shows | Backends |
|---|---|---|
| [luisa_ntc_cli](examples/luisa_ntc_cli.md) | Compress a material into an NTEX v3 neural texture, inspect an existing `.ntc` container, or decompress one back to PNG or BCn DDS — the training and conversion half of the NTC pipeline. | dx, vk, cuda |
| [luisa_ntc_bench](examples/luisa_ntc_bench.md) | Measures NTC decompression throughput per mip level and the speed of the GPU BCn transcoders on a `.ntc` container. | dx, vk, cuda |
| [luisa_ntc_test](examples/luisa_ntc_test.md) | Four-test verification suite for NTC: container round-trip, decode parity against reference images, BCn transcoding and a short training run with QAT export. | dx, vk, cuda |
| [luisa_ntc_viewer](examples/luisa_ntc_viewer.md) | Previews the channels of an NTC material — in a window, or offscreen into `viewer_output.png` for scripts and docs. | dx, vk, cuda |

## Summary

| Group | Pages |
|---|---|
| rendering | 20 |
| simulation | 6 |
| gui | 6 |
| compute | 12 |
| tensor | 1 |
| ml | 1 |
| coro | 3 |
| extension | 2 |
| dx | 1 |
| interop | 3 |
| embedding | 3 |
| embedding_mm | 1 |
| ocr | 2 |
| qwen | 1 |
| ntc | 4 |
| **total** | **66** |
