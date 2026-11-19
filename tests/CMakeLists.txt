# Tests moved here together with the examples they exercise (src/).
#
# Every target is declared through gallery_test() from cmake/gallery_examples.cmake
# (the CMake mirror of gallery_test() in scripts/gallery_build.lua). That helper
# already provides what LuisaCompute's local helper did: the src/ include root (so
# "embedding/embed/xxx.h" and "ocr/xxx.h" resolve), the test-support headers that
# stayed in LuisaCompute (${GALLERY_LC_TEST_INCLUDE_DIRS}: ut/ut.hpp,
# test_device.h) and the luisa::compute link.
#
# The target names are unchanged from LuisaCompute/src/tests/CMakeLists.txt, as are
# the CTest labels; keep this file and tests/xmake.lua in sync.

# ---------------------------------------------------------------------------
# Fixture roots compiled into the tests. The directories are local
# prerequisites and are not committed, see tests/data/README.md.
# ---------------------------------------------------------------------------
set(GALLERY_EMBED_TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/tests/data/embedding")
set(GALLERY_OCR_TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/tests/data/ocr")
set(GALLERY_OCR_EXAMPLES_DIR "${CMAKE_SOURCE_DIR}/src")

# <name>="<value>" as a single compile definition: the quotes are part of the
# macro value, exactly like LuisaCompute's own tests spelled them.
function(_gallery_string_define name value out_var)
    set(${out_var} "${name}=\"${value}\"" PARENT_SCOPE)
endfunction()

_gallery_string_define(LUISA_EMBED_TEST_DATA_DIR "${GALLERY_EMBED_TEST_DATA_DIR}"
        GALLERY_EMBED_TEST_DATA_DEFINE)
_gallery_string_define(LUISA_OCR_TEST_DATA_DIR "${GALLERY_OCR_TEST_DATA_DIR}"
        GALLERY_OCR_TEST_DATA_DEFINE)
_gallery_string_define(LUISA_OCR_EXAMPLES_DIR "${GALLERY_OCR_EXAMPLES_DIR}"
        GALLERY_OCR_EXAMPLES_DEFINE)

# ---------------------------------------------------------------------------
# unit/core: host-only tests for the plan/option helpers that moved into
# src/common (nbody_render_plan.h, path_tracing_sample_plan.h,
# photon_mapping_plan.h, reference_compare.h, coro_scheduler_options.h).
# They are included by bare name, hence the extra $SRC/common include dir.
# ---------------------------------------------------------------------------
foreach (_gallery_core_test IN ITEMS
        test_foreground_moments
        test_example_options
        test_path_tracing_sample_plan
        test_nbody_render_plan
        test_photon_mapping_plan)
    gallery_test(${_gallery_core_test}
            SOURCES "unit/core/${_gallery_core_test}.cpp"
            INCLUDES "$SRC/common"
            LABELS "unit;unit_core")
endforeach ()

# ---------------------------------------------------------------------------
# embedding engine (bge-m3): the examples live in src/embedding.
# ---------------------------------------------------------------------------
# host-only (LUISA_EMBED_HOST_ONLY=1): no device, no backend, no model.
set(_gallery_embed_host_only_tests
        test_embed_tokenizer
        test_embed_chunker
        test_embed_package
        test_embed_pack
        test_embed_tokenizer_mt
        test_embed_tokenizer_pool
        test_embed_retrieval_index
        test_embed_retrieval_rpc
        test_embed_retrieval_api)
foreach (_gallery_embed_test IN LISTS _gallery_embed_host_only_tests)
    set(_gallery_embed_deps "")
    set(_gallery_embed_includes "$SRC")
    if (_gallery_embed_test STREQUAL "test_embed_retrieval_api")
        # C-FFI shared library front-end (src/embedding/api): host-only, it drives
        # every exported step in callback mode, so it needs the api/ header.
        list(APPEND _gallery_embed_deps example_embed_retrieval_lib)
        list(APPEND _gallery_embed_includes "$SRC/embedding/api")
    endif ()
    gallery_test(${_gallery_embed_test}
            SOURCES "unit/embedding/${_gallery_embed_test}.cpp"
            DEPS lc-example-embedding ${_gallery_embed_deps}
            INCLUDES ${_gallery_embed_includes}
            DEFINES ${GALLERY_EMBED_TEST_DATA_DEFINE} LUISA_EMBED_HOST_ONLY=1
            LABELS "unit;unit_embedding")
endforeach ()
# retrieval_rpc talks TCP, so it needs Winsock on windows.
if (WIN32)
    target_link_libraries(test_embed_retrieval_rpc PRIVATE ws2_32)
endif ()

# Device tests: pass the backend as argv[1] when run by hand; CTest runs them on
# the fallback backend unless a GPU backend is selected via LUISA_EMBED_DEVICE_BACKEND.
set(LUISA_EMBED_DEVICE_BACKEND "fallback" CACHE STRING "Backend for embedding device tests")
set(_gallery_embed_device_tests
        test_embed_kernel_gemm
        test_embed_kernel_layernorm
        test_embed_kernel_embed
        test_embed_kernel_heads
        # Phase B (attention/executor/scheduler/debug)
        test_embed_kernel_attention
        test_embed_executor
        test_embed_scheduler
        test_embed_debug
        # test-plan wave (L1 ragged robustness, L2 e2e, L3 retrieval, L5 robustness)
        test_embed_kernel_robust
        test_embed_e2e
        test_embed_retrieval
        test_embed_robustness
        # Phase-0 tensor/coop probes (tensor_optimize.md Phase 0)
        test_embed_tensor_micro
        test_embed_coop_probe)
foreach (_gallery_embed_test IN LISTS _gallery_embed_device_tests)
    gallery_test(${_gallery_embed_test}
            SOURCES "integration/embedding/${_gallery_embed_test}.cpp"
            DEPS lc-example-embedding
            INCLUDES "$SRC"
            DEFINES ${GALLERY_EMBED_TEST_DATA_DEFINE}
            LABELS "integration;integration_embedding"
            ARGS "${LUISA_EMBED_DEVICE_BACKEND}")
    set_tests_properties(${_gallery_embed_test} PROPERTIES
            ENVIRONMENT "LUISA_ENABLE_VALIDATION=1")
endforeach ()

# ---------------------------------------------------------------------------
# OCR engine: the examples live in src/ocr. Both the fixture root and the
# examples root are compiled in, mirroring the embedding block above.
# ---------------------------------------------------------------------------
# host-only contract/artifact/output tests (no device, no model).
set(_gallery_ocr_host_only_tests
        test_ocr_recipe
        test_ocr_status
        test_ocr_cli
        test_ocr_quantization
        test_ocr_package
        test_ocr_safetensors
        test_ocr_tokenizer
        test_ocr_prompt
        test_ocr_image
        test_ocr_document
        test_ocr_preprocess
        test_ocr_json_output
        test_ocr_benchmark
        test_ocr_memory_budget
        test_ocr_runtime
        test_ocr_decoder_config
        test_ocr_output_table
        test_ocr_output_markdown
        test_ocr_output_extract
        test_ocr_model_contract
        test_ocr_ctc
        test_ocr_db
        test_ocr_reading_order
        test_ocr_cnn_result
        test_ocr_scheduler
        test_ocr_layout_decode
        test_ocr_layout_compose
        test_ocr_deepseek_contract)
# The OCR headers are included as "ocr/xxx.h", i.e. relative to the example group
# directory src/ocr (the way "embedding/xxx.h" is relative to src/).
foreach (_gallery_ocr_test IN LISTS _gallery_ocr_host_only_tests)
    gallery_test(${_gallery_ocr_test}
            SOURCES "unit/ocr/${_gallery_ocr_test}.cpp"
            DEPS lc-example-ocr
            INCLUDES "$SRC/ocr"
            DEFINES ${GALLERY_OCR_TEST_DATA_DEFINE} ${GALLERY_OCR_EXAMPLES_DEFINE}
            LABELS "unit;unit_ocr")
endforeach ()

# Device tests (integration;integration_ocr). They are always built, but a CTest
# entry is created ONLY when LUISA_OCR_DEVICE_BACKEND names a backend, so an empty
# value never produces a silently-skipping test. Run them manually with an
# explicit backend argument otherwise: test_ocr_ops_gemm vk|cuda|fallback.
set(LUISA_OCR_DEVICE_BACKEND "" CACHE STRING
        "Backend argument for the OCR device tests' CTest entries (empty = build only, no CTest entry)")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
set(_gallery_ocr_device_tests
        test_ocr_ops_image
        test_ocr_ops_gemm
        test_ocr_ops_norm
        test_ocr_ops_rope
        test_ocr_ops_attention
        test_ocr_ops_conv
        test_ocr_ops_vision
        # Phase 9 GDN operators (synthetic)
        test_ocr_gdn_conv
        test_ocr_gdn_recurrent
        test_ocr_gdn_chunked
        test_ocr_gdn_cache
        # Phase 9 model vertical slices (synthetic)
        test_ocr_model_ovis_ocr2
        test_ocr_model_surya
        # Phase 10 DeepSeek-OCR-2 MoE vertical slice (synthetic)
        test_ocr_moe_router
        test_ocr_moe_permutation
        test_ocr_moe_experts
        test_ocr_model_deepseek_ocr2
        test_ocr_runtime_device
        # Phase 3 dense decoder
        test_ocr_decode_prefill
        test_ocr_decode_cache
        test_ocr_decode_sampling
        test_ocr_decode_stop
        # Phase 4 Qwen3-VL vertical slice + FireRed-OCR
        test_ocr_model_qwen3_vl
        test_ocr_model_fire_red
        # Phase 5 synthetic PP-OCRv6 CNN pipeline
        test_ocr_model_pp_ocrv6
        # Phase 6 layout/region infrastructure + PaddleOCR-VL + OpenDoc
        test_ocr_layout
        test_ocr_model_paddle_ocr_vl
        test_ocr_model_open_doc
        test_ocr_region_scheduler
        # Phase 7 custom dense/layout variants (synthetic)
        test_ocr_model_tele_ocr
        test_ocr_model_glm_ocr
        test_ocr_model_mineru
        test_ocr_model_hunyuan
        # Phase 8 remaining dense ViT models (synthetic)
        test_ocr_model_dots_mocr
        test_ocr_model_got_ocr2
        test_ocr_model_olm_ocr2
        # benchmark smoke test (reuses the separate benchmark-support library)
        test_ocr_benchmark_device)
foreach (_gallery_ocr_test IN LISTS _gallery_ocr_device_tests)
    set(_gallery_ocr_deps "")
    if (_gallery_ocr_test STREQUAL "test_ocr_benchmark_device")
        list(APPEND _gallery_ocr_deps lc-example-ocr-bench)
    endif ()
    # "ocr/xxx.h" resolves through src/ocr, the shared ocr_*_test_utils.h headers
    # sit next to the sources.
    gallery_test(${_gallery_ocr_test} NO_CTEST
            SOURCES "integration/ocr/${_gallery_ocr_test}.cpp"
            DEPS lc-example-ocr ${_gallery_ocr_deps}
            INCLUDES "$SRC/ocr" "integration/ocr"
            DEFINES ${GALLERY_OCR_TEST_DATA_DEFINE} ${GALLERY_OCR_EXAMPLES_DEFINE})
    if (LUISA_OCR_DEVICE_BACKEND)
        add_test(NAME ${_gallery_ocr_test}
                COMMAND $<TARGET_FILE:${_gallery_ocr_test}> ${LUISA_OCR_DEVICE_BACKEND})
        set_tests_properties(${_gallery_ocr_test} PROPERTIES
                LABELS "integration;integration_ocr"
                TIMEOUT 900
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
    endif ()
endforeach ()

# ---------------------------------------------------------------------------
# Qwen3.5-4B engine (src/qwen): host-only / device-light unit tests. No device,
# backend or weights are required; the real-GGUF tokenizer test reads
# data/Qwen3.5-4B-BF16.gguf relative to the working directory when present.
foreach (_gallery_qwen_test IN ITEMS
        test_qwen_gguf
        test_qwen_weights
        test_qwen_engine
        test_qwen_speculative)
    gallery_test(${_gallery_qwen_test}
            SOURCES "unit/qwen/${_gallery_qwen_test}.cpp"
            DEPS lc-example-qwen
            INCLUDES "$SRC/qwen" "$SRC/ocr"
            LABELS "unit;unit_qwen")
endforeach ()

# Qwen device operator regression tests (tests/integration/qwen): synthetic
# data only (no checkpoint), explicit backend argument. Built always; CTest
# entries are added only when a backend is selected via LUISA_QWEN_DEVICE_BACKEND
# (the tests fail without an explicit backend, so no default is registered).
# cmake -DLUISA_QWEN_DEVICE_BACKEND=cuda ...   # dsl + tile
set(LUISA_QWEN_DEVICE_BACKEND "" CACHE STRING
        "Backend argument for the Qwen device tests' CTest entries (empty = build only, no CTest entry)")
foreach (_gallery_qwen_device_test IN ITEMS
        test_qwen_ops_dsl
        test_qwen_ops_tile
        test_qwen_mtp)
    gallery_test(${_gallery_qwen_device_test} NO_CTEST
            SOURCES "integration/qwen/${_gallery_qwen_device_test}.cpp"
            DEPS lc-example-qwen
            INCLUDES "$SRC/qwen" "$SRC/ocr" "integration/qwen")
    if (LUISA_QWEN_DEVICE_BACKEND)
        add_test(NAME ${_gallery_qwen_device_test}
                COMMAND $<TARGET_FILE:${_gallery_qwen_device_test}> ${LUISA_QWEN_DEVICE_BACKEND})
        set_tests_properties(${_gallery_qwen_device_test} PROPERTIES
                LABELS "integration;integration_qwen"
                TIMEOUT 900
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")
    endif ()
endforeach ()

# ---------------------------------------------------------------------------
# Benchmarks (built, not part of the test suite -- as in LuisaCompute, where they
# were registered without LABELS and therefore never ran under CTest).
# ---------------------------------------------------------------------------
# Optimization-phase benchmark for the bge-m3 embedding engine
# (embedding_model_opt_plan.md O0.1; needs the packed model via --model/BGEM3_MODEL_DIR).
gallery_test(benchmark_embed NO_CTEST
        SOURCES "benchmark/benchmark_embed.cpp"
        DEPS lc-example-embedding
        INCLUDES "$SRC" "$SRC/embedding")

# GEMM-family tile sweep for the bge-m3 embedding engine
# (embedding_model_opt_plan.md §3.1 G1/G2/G3/G4/G8; synthetic shapes only,
# no packed model needed): benchmark_embed_gemm <dx|vk|fallback> [--rows 512,4096] [--csv f.csv]
gallery_test(benchmark_embed_gemm NO_CTEST
        SOURCES "benchmark/benchmark_embed_gemm.cpp"
        DEPS lc-example-embedding
        INCLUDES "$SRC" "$SRC/embedding")
