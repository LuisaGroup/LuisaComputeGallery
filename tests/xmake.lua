-- Tests moved here together with the examples they exercise (src/).
--
-- Every target is declared through gallery_test() from scripts/gallery_build.lua,
-- the xmake mirror of gallery_test() in cmake/gallery_examples.cmake. That helper
-- already provides what LuisaCompute's local `test_proj()` did: the src/ include
-- root (so "embedding/embed/xxx.h" and "ocr/xxx.h" resolve), the test-support
-- headers that stayed in LuisaCompute ($LC/src/tests, $LC/src/tests/common, for
-- ut/ut.hpp and test_device.h) and the lc-runtime/lc-dsl/lc-vstl/stb-image module
-- dependencies every test needs.
--
-- The target names are unchanged from LuisaCompute/src/tests/{xmake.lua,CMakeLists.txt};
-- keep this file and tests/CMakeLists.txt in sync (same targets, same defines).

-- Fixture roots are compiled into the tests (absolute, forward-slash paths).
-- The directories are local prerequisites and are not committed, see
-- tests/data/README.md.
local function gallery_test_data_dir(name)
    return path.unix(path.join(os.projectdir(), "tests", "data", name))
end

-- <name>="<value>" as one compile definition.
local function gallery_string_define(name, value)
    return string.format('%s="%s"', name, value)
end

-- ---------------------------------------------------------------------------
-- unit/core: host-only tests for the plan/option helpers that moved into
-- src/common (nbody_render_plan.h, path_tracing_sample_plan.h,
-- photon_mapping_plan.h, reference_compare.h, coro_scheduler_options.h).
-- ---------------------------------------------------------------------------
local core_tests = {
    "test_foreground_moments",
    "test_example_options",
    "test_path_tracing_sample_plan",
    "test_nbody_render_plan",
    "test_photon_mapping_plan",
}
for _, name in ipairs(core_tests) do
    gallery_test(name, {
        sources = {"unit/core/" .. name .. ".cpp"},
        -- the helpers are included by bare name ("reference_compare.h"), so the
        -- moved examples/common directory itself has to be on the search path
        includes = {"$SRC/common"},
    })
end

-- ---------------------------------------------------------------------------
-- embedding engine (bge-m3): the examples live in src/embedding.
-- ---------------------------------------------------------------------------
local embed_test_define = gallery_string_define("LUISA_EMBED_TEST_DATA_DIR",
    gallery_test_data_dir("embedding"))

-- Host-only (LUISA_EMBED_HOST_ONLY=1): no device, no backend, no model.
local embed_host_only_tests = {
    "test_embed_tokenizer",
    "test_embed_chunker",
    "test_embed_package",
    "test_embed_pack",
    "test_embed_tokenizer_mt",
    "test_embed_tokenizer_pool",
    "test_embed_retrieval_index",
    "test_embed_retrieval_rpc",
    "test_embed_retrieval_api",
}
for _, name in ipairs(embed_host_only_tests) do
    local options = {
        sources = {"unit/embedding/" .. name .. ".cpp"},
        deps = {"lc-example-embedding"},
        includes = {"$SRC"},
        defines = {embed_test_define, "LUISA_EMBED_HOST_ONLY=1"},
    }
    -- retrieval_rpc talks TCP, so it needs Winsock on windows.
    if name == "test_embed_retrieval_rpc" and is_plat("windows") then
        options.syslibs = {"ws2_32"}
    end
    -- C-FFI shared-library front-end (src/embedding/api): it drives the exported
    -- steps in callback mode, so the header must come from the api/ directory.
    if name == "test_embed_retrieval_api" then
        table.insert(options.deps, "example_embed_retrieval_lib")
        table.insert(options.includes, "$SRC/embedding/api")
    end
    gallery_test(name, options)
end

-- Device tests: pass the backend as argv[1] when running them by hand.
local embed_device_tests = {
    "test_embed_kernel_gemm",
    "test_embed_kernel_layernorm",
    "test_embed_kernel_embed",
    "test_embed_kernel_heads",
    -- Phase B (attention/executor/scheduler/debug)
    "test_embed_kernel_attention",
    "test_embed_executor",
    "test_embed_scheduler",
    "test_embed_debug",
    -- test-plan wave (L1 ragged robustness, L2 e2e, L3 retrieval, L5 robustness)
    "test_embed_kernel_robust",
    "test_embed_e2e",
    "test_embed_retrieval",
    "test_embed_robustness",
    -- Phase-0 tensor/coop probes (tensor_optimize.md Phase 0)
    "test_embed_tensor_micro",
    "test_embed_coop_probe",
}
for _, name in ipairs(embed_device_tests) do
    gallery_test(name, {
        sources = {"integration/embedding/" .. name .. ".cpp"},
        deps = {"lc-example-embedding"},
        includes = {"$SRC"},
        defines = {embed_test_define},
    })
end

-- ---------------------------------------------------------------------------
-- OCR engine: the examples live in src/ocr. Both the fixture root and the
-- examples root are compiled in, mirroring the embedding helper above.
-- ---------------------------------------------------------------------------
local ocr_defines = {
    gallery_string_define("LUISA_OCR_TEST_DATA_DIR", gallery_test_data_dir("ocr")),
    gallery_string_define("LUISA_OCR_EXAMPLES_DIR", path.unix(path.join(os.projectdir(), "src"))),
}

-- Host-only contract/artifact/output tests (no device, no model).
local ocr_host_only_tests = {
    "test_ocr_recipe",
    "test_ocr_status",
    "test_ocr_cli",
    "test_ocr_quantization",
    "test_ocr_package",
    "test_ocr_safetensors",
    "test_ocr_tokenizer",
    "test_ocr_prompt",
    "test_ocr_image",
    "test_ocr_document",
    "test_ocr_preprocess",
    "test_ocr_json_output",
    "test_ocr_benchmark",
    "test_ocr_memory_budget",
    "test_ocr_runtime",
    "test_ocr_decoder_config",
    "test_ocr_output_table",
    "test_ocr_output_markdown",
    "test_ocr_output_extract",
    "test_ocr_model_contract",
    "test_ocr_ctc",
    "test_ocr_db",
    "test_ocr_reading_order",
    "test_ocr_cnn_result",
    "test_ocr_scheduler",
    "test_ocr_layout_decode",
    "test_ocr_layout_compose",
    "test_ocr_deepseek_contract",
}
for _, name in ipairs(ocr_host_only_tests) do
    gallery_test(name, {
        sources = {"unit/ocr/" .. name .. ".cpp"},
        deps = {"lc-example-ocr"},
        -- the OCR headers are included as "ocr/xxx.h", i.e. relative to the
        -- example group directory src/ocr (as "embedding/xxx.h" is to src/)
        includes = {"$SRC/ocr"},
        defines = ocr_defines,
    })
end

-- Device tests: run with an explicit backend, e.g.
--   xmake run test_ocr_ops_gemm vk|cuda|fallback
local ocr_device_tests = {
    "test_ocr_ops_image",
    "test_ocr_ops_gemm",
    "test_ocr_ops_norm",
    "test_ocr_ops_rope",
    "test_ocr_ops_attention",
    "test_ocr_ops_conv",
    "test_ocr_ops_vision",
    -- Phase 9 GDN operators (synthetic)
    "test_ocr_gdn_conv",
    "test_ocr_gdn_recurrent",
    "test_ocr_gdn_chunked",
    "test_ocr_gdn_cache",
    -- Phase 9 model vertical slices (synthetic)
    "test_ocr_model_ovis_ocr2",
    "test_ocr_model_surya",
    -- Phase 10 DeepSeek-OCR-2 MoE vertical slice (synthetic)
    "test_ocr_moe_router",
    "test_ocr_moe_permutation",
    "test_ocr_moe_experts",
    "test_ocr_model_deepseek_ocr2",
    "test_ocr_runtime_device",
    -- Phase 3 dense decoder
    "test_ocr_decode_prefill",
    "test_ocr_decode_cache",
    "test_ocr_decode_sampling",
    "test_ocr_decode_stop",
    -- Phase 4 Qwen3-VL vertical slice + FireRed-OCR
    "test_ocr_model_qwen3_vl",
    "test_ocr_model_fire_red",
    -- Phase 5 synthetic PP-OCRv6 CNN pipeline
    "test_ocr_model_pp_ocrv6",
    -- Phase 6 layout/region infrastructure + PaddleOCR-VL + OpenDoc
    "test_ocr_layout",
    "test_ocr_model_paddle_ocr_vl",
    "test_ocr_model_open_doc",
    "test_ocr_region_scheduler",
    -- Phase 7 custom dense/layout variants (synthetic)
    "test_ocr_model_tele_ocr",
    "test_ocr_model_glm_ocr",
    "test_ocr_model_mineru",
    "test_ocr_model_hunyuan",
    -- Phase 8 remaining dense ViT models (synthetic)
    "test_ocr_model_dots_mocr",
    "test_ocr_model_got_ocr2",
    "test_ocr_model_olm_ocr2",
    "test_ocr_benchmark_device",
}
for _, name in ipairs(ocr_device_tests) do
    local options = {
        sources = {"integration/ocr/" .. name .. ".cpp"},
        deps = {"lc-example-ocr"},
        -- "ocr/xxx.h" resolves through the example group dir, the shared
        -- ocr_*_test_utils.h headers sit next to the sources
        includes = {"$SRC/ocr", "integration/ocr"},
        defines = ocr_defines,
    }
    -- the benchmark smoke test reuses the separate benchmark-support library
    if name == "test_ocr_benchmark_device" then
        table.insert(options.deps, "lc-example-ocr-bench")
    end
    gallery_test(name, options)
end

-- ---------------------------------------------------------------------------
-- Benchmarks (built, not part of the test suite).
-- ---------------------------------------------------------------------------
-- Optimization-phase benchmark for the bge-m3 embedding engine
-- (embedding_model_opt_plan.md O0.1); run with
--   xmake run benchmark_embed <backend> --model <dir> [--seq N] [--batch B] ...
gallery_test("benchmark_embed", {
    sources = {"benchmark/benchmark_embed.cpp"},
    deps = {"lc-example-embedding"},
    -- includes "embed/session.h" & friends relative to the embedding example
    includes = {"$SRC", "$SRC/embedding"},
})

-- GEMM-family tile sweep for the bge-m3 embedding engine
-- (embedding_model_opt_plan.md §3.1 G1/G2/G3/G4/G8); synthetic shapes only, no
-- packed model needed. Run with
--   xmake run benchmark_embed_gemm <backend> [--rows 512,4096] [--csv out.csv]
gallery_test("benchmark_embed_gemm", {
    sources = {"benchmark/benchmark_embed_gemm.cpp"},
    deps = {"lc-example-embedding"},
    includes = {"$SRC", "$SRC/embedding"},
})
