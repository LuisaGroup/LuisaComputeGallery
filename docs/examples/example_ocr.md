# example_ocr

Command-line front end of the LuisaCompute OCR engine: validates a complete recognition request, prints the model support report offline, and refuses unvalidated models instead of inventing text.

| | |
|---|---|
| Group | `ocr` |
| Sources | `src/ocr/example_ocr.cpp`, `src/ocr/ocr/cli.cpp`, `src/ocr/ocr/api.h`, `src/ocr/ocr/model_registry.cpp`, `src/ocr/ocr/session.cpp`, `src/ocr/ocr/output/json_output.cpp` |
| Backends | none (offline tool, no device) |
| Build | `xmake build example_ocr` |
| Run | `example_ocr --list-models` |

## Purpose

The engine surface of the OCR family: `example_ocr` parses and validates the full recognition request (backend, package, pages, mode, precision, cache and memory policies, sampling, pixel/token budgets), resolves it against the model registry, and then fails closed. Two entry points work on any machine with no GPU and no package — `--help` and `--list-models` — and the second one is the machine-readable answer to "which OCR models does this engine actually support?".

## Architecture

- Offline surface. `--list-models` builds the `ModelRegistry` and prints its 17 selectable targets as one JSON object with `target`, `model`, `family` (`cnn`, `dense_vlm`, `layout_vlm`, `hybrid_gdn`, `moe_vlm`, `custom_vlm`), `support_state` and `validated`; `target_count`, `real_model_validated` and `session_status` summarise the table. It runs before any device or package work.
- Request validation. `parse_cli_args` (Windows: `wmain` plus a CP_UTF8 conversion, so non-ASCII paths and prompts survive) fills a `CliArgs`; `validate_session_config` checks the backend/package/manifest combination and `Session::preflight_request` checks the pages and the `RecognizeOptions` (mode `plain`/`spotting`/`markdown`/`extract`, prompt and schema for extract, language and region/colour guidance, `automatic|fp32|fp16|bf16|int8|int4` precisions, `contiguous|paged` cache, graph and fast-path policies, device/host/wall-time budgets). The resolved configuration is echoed to stderr, including the effective overview/crop pixel bounds and the per-region/page token ceilings.
- Fail-closed behaviour. `Session::create` accepts no real artifact today: the call is refused with status `unvalidated`, and the refusal is serialised as structured JSON — never as OCR text. A successful `recognize()` would be written through `output::write_result` in the full `RecognizeResult` schema (canonical text, lines and blocks with source coordinates, confidence-presence flags, metadata, diagnostics, counters).
- Support matrix. `src/ocr/MODELS.md` records what each recipe has been exercised against and which official gate is still unmet; `QUALIFICATION.md` holds the per-target qualification matrix. Synthetic operator parity is deliberately *not* counted as model support.

## Usage

Nothing from `data/` is involved here — there is no `python scripts/unpack_data.py` step, because the repository ships no OCR package; `--help` and `--list-models` need neither a device nor a package.

```bash
example_ocr --help                 # usage text, no device, exit 0
example_ocr --list-models          # support report JSON, no device, exit 0
example_ocr --backend dx --package ocr.lcocr --input page.png --mode markdown
example_ocr --backend vk --package ocr.lcocr --input page1.png --input page2.png --model pp_ocrv6:medium --max-output-tokens-per-page 4096
```

| Option | Meaning |
|---|---|
| `--help`, `--list-models` | offline: usage text or the support report, exit 0 |
| `--backend <name>` | backend plugin (`dx`, `vk`, `cuda`, `fallback`, ...); the name is validated and echoed, but this engine refuses before a device is created |
| `--package <file.lcocr>`, `--manifest <file.json>` | packed model package and/or its component manifest (a validated `.lcocr` package is required for real inference) |
| `--input <image>` | input page, repeatable, in reading order |
| `--model <target>` | recipe name, e.g. `pp_ocrv6:tiny`, `paddle_ocr_vl_1_6`, `dots_mocr`, `olm_ocr_2` |
| `--mode plain\|spotting\|markdown\|extract` | output mode; `--prompt`, `--schema`, `--language`, `--region-guidance`, `--color-guidance` feed the prompt-directed modes |
| `--output <path>` | result or status JSON file (default stdout) |
| `--temperature`, `--top-p`, `--top-k`, `--seed` | sampling; greedy unless `--temperature > 0` |
| `--max-overview-pixels`, `--max-crop-pixels`, `--max-pixels` | image bounds; `--max-output-tokens-per-region/-page`, `--max-regions-per-batch`, `--max-decode-batch` bound the decoder |
| `--weight-precision`, `--activation-precision`, `--cache-precision`, `--cache-policy`, `--offload`, `--graph`, `--fast-path` | numerical and execution policies (`automatic` by default) |
| `--max-device-bytes`, `--max-host-bytes`, `--max-wall-time-ms` | resource budgets |
| `--unsafe-experimental` | explicit override for unknown signatures; the target stays unvalidated |
| `--headless` | same work, no window; this tool never opens one |

## Result

`--list-models` prints one JSON object (formatted here for reading, 15 entries elided):

```json
{"target_count":17,"real_model_validated":false,"session_status":"unvalidated","models":[
  {"target":"pp_ocrv6:tiny","model":"pp_ocrv6","family":"cnn","support_state":"planned","validated":false},
  {"target":"pp_ocrv6:small","model":"pp_ocrv6","family":"cnn","support_state":"planned","validated":false},
  ... all 17 entries report support_state "planned" and validated false ...]}
```

A run that asks for real inference reports its resolved configuration and the refusal, then exits 2:

```text
example_ocr: backend=dx model=(from manifest) mode=plain precision=automatic/automatic/automatic overview_pixels=0 crop_pixels=0 tokens_region=2048 tokens_page=8192
{"ok":false,"code":"unvalidated","message":"no OCR model is validated yet: session creation is refused; see examples/ocr/MODELS.md","detail":"model"}
```

Exit codes: 0 for `--help`/`--list-models`, 1 for invalid arguments, 2 for an engine refusal, 3 when the report cannot be written.

## See also

example_ocr_bench (correctness-gated benchmark of the same engine), example_embed_text (a model-backed engine that does run end to end).
