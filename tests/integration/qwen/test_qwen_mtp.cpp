// Real-trunk MTP executor differential test. The NextN draft is an explicit
// synthetic all-zero fixture (never a model-quality or acceleration claim):
// it exercises the actual DSL draft kernels, target verify, acceptance,
// snapshot/restore and accepted-prefix replay against ordinary target decode.
#include "qwen_device_test_utils.h"

#include "qwen/engine.h"
#include "qwen/gguf.h"
#include "qwen/mtp_executor.h"
#include "qwen/weights.h"

#include <luisa/core/stl/filesystem.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::qwen;
using namespace qwen_test;

namespace {

template<typename T>
[[nodiscard]] Buffer<T> zero_buffer(Device &device, Stream &stream, uint64_t count) {
    auto buffer = device.create_buffer<T>(static_cast<size_t>(count));
    if (!buffer || count == 0u) { return buffer; }
    constexpr size_t chunk = 1u << 18u;
    luisa::vector<T> zeros(std::min<uint64_t>(count, chunk), T{});
    for (uint64_t done = 0; done < count;) {
        const auto n = static_cast<size_t>(std::min<uint64_t>(chunk, count - done));
        // Immediate stream encoding plus the synchronize before `zeros` dies
        // keeps the borrowed host span valid on every backend.
        stream << buffer.view(static_cast<size_t>(done), n).copy_from(
            luisa::span<const T>{zeros.data(), n});
        done += n;
    }
    stream.synchronize();
    return buffer;
}

[[nodiscard]] QwenDeviceTensor zero_tensor(
    Device &device, Stream &stream, const char *name,
    luisa::vector<uint64_t> dims, bool dense, bool fp8 = false) {
    QwenDeviceTensor tensor;
    tensor.role = QwenTensorRole::gdn_qkv;
    tensor.layer = 32u;
    tensor.name = name;
    tensor.dims = std::move(dims);
    tensor.element_count = 1u;
    for (const auto d : tensor.dims) { tensor.element_count *= d; }
    if (dense) {
        tensor.storage = WeightStorage::f32_dense;
        tensor.f32 = zero_buffer<float>(device, stream, tensor.element_count);
    } else if (fp8) {
        tensor.storage = WeightStorage::fp8_packed;
        tensor.packed = zero_buffer<uint32_t>(device, stream, (tensor.element_count + 3u) / 4u);
        tensor.groups_per_row = (tensor.dims[0] + 63u) / 64u;
        tensor.scales = zero_buffer<float>(device, stream, tensor.groups_per_row * tensor.dims[1]);
    } else {
        tensor.storage = WeightStorage::bf16_packed;
        tensor.packed = zero_buffer<uint32_t>(device, stream, (tensor.element_count + 1u) / 2u);
    }
    return tensor;
}

[[nodiscard]] QwenDeviceWeights make_zero_trunk(
    Device &device, Stream &stream, const Qwen35Model &model, bool fp8 = false) {
    luisa::vector<QwenDeviceTensor> tensors;
    for (const auto &ref : model.tensors()) {
        const auto &info = *ref.info;
        auto storage = classify_weight_storage(info.name, info.type, info.dims, false);
        if (!storage.ok()) { continue; }
        auto tensor = zero_tensor(device, stream, info.name.c_str(), info.dims,
                                  *storage.value == WeightStorage::f32_dense, fp8);
        tensor.role = ref.role;
        tensor.layer = ref.layer;
        tensors.emplace_back(std::move(tensor));
    }
    return QwenDeviceWeights::from_device_tensors(std::move(tensors));
}

[[nodiscard]] QwenDeviceWeights make_zero_draft(
    Device &device, Stream &stream, const Qwen35Model &model) {
    const auto &g = model.config().geometry;
    luisa::vector<QwenDeviceTensor> tensors;
    const auto add = [&](const char *suffix, luisa::vector<uint64_t> dims, bool dense = false) {
        const auto name = luisa::string{"blk.32."} + suffix;
        tensors.emplace_back(zero_tensor(device, stream, name.c_str(), std::move(dims), dense));
    };
    add("eh_proj.weight", {2u * g.hidden, g.hidden});
    add("enorm.weight", {g.hidden}, true);
    add("hnorm.weight", {g.hidden}, true);
    add("shared_head.norm.weight", {g.hidden}, true);
    add("attn_norm.weight", {g.hidden}, true);
    add("post_attention_norm.weight", {g.hidden}, true);
    add("attn_q.weight", {g.hidden, g.q_proj_n});
    add("attn_k.weight", {g.hidden, g.kv_proj_n});
    add("attn_v.weight", {g.hidden, g.kv_proj_n});
    add("attn_output.weight", {g.o_proj_k, g.hidden});
    add("attn_q_norm.weight", {g.head_dim}, true);
    add("attn_k_norm.weight", {g.head_dim}, true);
    add("ffn_gate.weight", {g.hidden, g.ffn});
    add("ffn_up.weight", {g.hidden, g.ffn});
    add("ffn_down.weight", {g.ffn, g.hidden});
    return QwenDeviceWeights::from_device_tensors(std::move(tensors));
}

// FP8 engine compile + four-step in-flight graph lifecycle probe. The GGUF is
// used only for its validated geometry/inventory; every weight is zero. This
// avoids the expensive host quantizer and makes no model-quality claim.
[[nodiscard]] int run_zero_graph(Device &device, Stream &stream, const Qwen35Model &model,
                                 bool fp8) {
    auto weights = make_zero_trunk(device, stream, model, fp8);
    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 32u;
    options.prefill_chunk = 8u;
    options.dsl_decode_specialized = true;
    options.dsl_cuda_graph = true;
    auto engine = QwenEngine::create(device, stream, model, weights, options);
    check(engine.ok(), "create zero-weight graph engine");
    if (!engine.ok()) { return summarize("qwen FP8 graph probe", "cuda"); }
    const luisa::vector<uint32_t> prompt(8u, 0u);
    luisa::vector<float> logits;
    check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits).ok(),
          "zero-weight graph prefill");
    uint32_t token = 0u;
    for (uint32_t step = 0; step < 4u; ++step) {
        const auto status = (*engine.value)->decode_device_greedy(
            stream, step == 0u ? &token : nullptr, step);
        check(status.ok(), "zero-weight graph decode step");
        if (!status.ok()) { return summarize("qwen FP8 graph probe", "cuda"); }
    }
    luisa::vector<uint32_t> history(4u, UINT32_MAX);
    check((*engine.value)->download_device_greedy(stream, 0u, luisa::span<uint32_t>{history}).ok(),
          "zero-weight graph history download");
    check(history == luisa::vector<uint32_t>(4u, 0u), "zero graph deterministic history");
    const auto info = (*engine.value)->info();
    // Static graph (opt_plan/03 section 2.3): the decode body is captured ONCE
    // and every later token is a bare launch, so the launch counter equals the
    // number of steps and the update counter is exactly zero.
    check(info.cuda_graph_steps == 4u, "graph launch counter == steps");
    check(info.cuda_graph_updates == 0u, "graph update counter == 0");
    check(info.position == 12u, "graph probe position accounting");
    return summarize(fp8 ? "qwen FP8 static graph probe" : "qwen BF16 static graph probe", "cuda");
}

// ---------------------------------------------------------------------------
// Static-graph A/B on the REAL trunk (CUDA only): graph on versus graph off must
// generate identical greedy ids, the graph arm must report zero whole-graph
// updates, and a reset + re-prefill must reuse the captured graph without
// corrupting the device decode state. The 504-token prompt puts the decode
// positions across the 512-token KV page boundary, which is the case where the
// graph-external ensure_capacity path is exercised.
// ---------------------------------------------------------------------------
struct GraphArm {
    luisa::vector<uint32_t> ids;
    luisa::vector<uint32_t> ids_after_reset;
    luisa::vector<uint32_t> mixed_ids;
    uint64_t steps{0};
    uint64_t updates{0};
    uint32_t position{0};
};

[[nodiscard]] int run_graph_ab(Device &device, Stream &stream, const Qwen35Model &model,
                               const QwenDeviceWeights &weights) {
    const auto &g = model.config().geometry;
    luisa::vector<uint32_t> prompt(504u);
    uint64_t lcg = 12345u;
    for (auto &id : prompt) {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        id = static_cast<uint32_t>((lcg >> 33u) % g.vocab);
    }
    constexpr uint32_t steps = 64u;
    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 1024u;
    options.prefill_chunk = 64u;
    options.dsl_decode_specialized = true;
    const auto run_arm = [&](bool graph_on) {
        GraphArm arm;
        options.dsl_cuda_graph = graph_on;
        auto engine = QwenEngine::create(device, stream, model, weights, options);
        check(engine.ok(), graph_on ? "graph A/B: create graph engine" : "graph A/B: create list engine");
        if (!engine.ok()) { return arm; }
        luisa::vector<float> logits;
        check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits).ok(),
              "graph A/B: prefill");
        auto next = qwen_argmax(logits);
        check(next.ok(), "graph A/B: host argmax");
        if (!next.ok()) { return arm; }
        uint32_t pending = *next.value;
        bool need_host = true;
        // Bounded in-flight chunks, exactly like the CLI decode policy.
        for (uint32_t done = 0; done < steps;) {
            const uint32_t count = std::min(4u, steps - done);
            for (uint32_t j = 0; j < count; j++) {
                const auto status = (*engine.value)->decode_device_greedy(
                    stream, need_host ? &pending : nullptr, done + j);
                check(status.ok(), "graph A/B: device greedy step");
                if (!status.ok()) { return arm; }
                need_host = false;
            }
            luisa::vector<uint32_t> chunk(count, UINT32_MAX);
            check((*engine.value)->download_device_greedy(stream, done, luisa::span<uint32_t>{chunk}).ok(),
                  "graph A/B: device greedy download");
            for (auto id : chunk) { arm.ids.emplace_back(id); }
            done += count;
        }
        {
            const auto info = (*engine.value)->info();
            arm.steps = info.cuda_graph_steps;
            arm.updates = info.cuda_graph_updates;
            arm.position = info.position;
        }
        // Mixed host-logits -> device-greedy transition. decode() advances the
        // position on the HOST-scalar path without touching the device decode
        // state, so the next static-graph launch must re-upload it; a stale
        // position would silently decode at the wrong RoPE offset and conv ring
        // index (no crash, wrong ids). Both arms must agree.
        {
            uint32_t token = arm.ids.empty() ? 0u : arm.ids.back();
            luisa::vector<float> host_logits;
            for (uint32_t j = 0; j < 2u; j++) {
                const auto status = (*engine.value)->decode(stream, token, host_logits);
                check(status.ok(), "graph A/B: host-logits decode");
                if (!status.ok()) { return arm; }
                auto arg = qwen_argmax(host_logits);
                check(arg.ok(), "graph A/B: host argmax after a host-logits decode");
                if (!arg.ok()) { return arm; }
                token = *arg.value;
                arm.mixed_ids.emplace_back(token);
            }
            for (uint32_t j = 0; j < 4u; j++) {
                const auto status = (*engine.value)->decode_device_greedy(
                    stream, j == 0u ? &token : nullptr, j);
                check(status.ok(), "graph A/B: device greedy step after a host-logits decode");
                if (!status.ok()) { return arm; }
            }
            luisa::vector<uint32_t> chunk(4u, UINT32_MAX);
            check((*engine.value)->download_device_greedy(stream, 0u, luisa::span<uint32_t>{chunk}).ok(),
                  "graph A/B: download after a host-logits decode");
            for (auto id : chunk) { arm.mixed_ids.emplace_back(id); }
        }
        // Reset + re-prefill: the captured graph must be reusable and the device
        // decode state must be re-uploaded for the new session.
        luisa::vector<uint32_t> short_prompt(prompt.begin(), prompt.begin() + 8);
        check((*engine.value)->reset(stream).ok(), "graph A/B: reset");
        luisa::vector<float> short_logits;
        check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{short_prompt}, short_logits).ok(),
              "graph A/B: re-prefill after reset");
        auto again = qwen_argmax(short_logits);
        check(again.ok(), "graph A/B: host argmax after reset");
        if (!again.ok()) { return arm; }
        uint32_t token = *again.value;
        for (uint32_t j = 0; j < 8u; j++) {
            const auto status = (*engine.value)->decode_device_greedy(stream, j == 0u ? &token : nullptr, j);
            check(status.ok(), "graph A/B: device greedy step after reset");
            if (!status.ok()) { return arm; }
        }
        luisa::vector<uint32_t> after(8u, UINT32_MAX);
        check((*engine.value)->download_device_greedy(stream, 0u, luisa::span<uint32_t>{after}).ok(),
              "graph A/B: download after reset");
        arm.ids_after_reset = std::move(after);
        std::printf("graph A/B arm: requested=%s steps=%llu updates=%llu position=%u ids=%s\n",
                    graph_on ? "static" : "list",
                    static_cast<unsigned long long>(arm.steps),
                    static_cast<unsigned long long>(arm.updates), arm.position,
                    arm.ids.empty() ? "-" : "ok");
        return arm;
    };
    auto graphed = run_arm(true);
    auto listed = run_arm(false);
    check(graphed.ids.size() == steps && listed.ids.size() == steps, "graph A/B: 64 greedy ids");
    check(!graphed.ids.empty() && graphed.ids == listed.ids,
          "graph A/B: graph and list ids are identical (64 tokens across a KV page boundary)");
    check(graphed.ids_after_reset == listed.ids_after_reset,
          "graph A/B: ids after reset + re-prefill are identical");
    check(graphed.steps == steps && graphed.updates == 0u,
          "graph A/B: one launch per step and zero graph updates");
    check(listed.steps == 0u && listed.updates == 0u, "graph A/B: list arm never captures a graph");
    check(graphed.position == 504u + steps, "graph A/B: host position accounting");
    check(!graphed.mixed_ids.empty() && graphed.mixed_ids == listed.mixed_ids,
          "graph A/B: ids identical across the host-logits -> device-greedy transition");
    return summarize("qwen graph A/B", "cuda");
}

// ---------------------------------------------------------------------------
// Split-KV decode attention A/B on the REAL trunk (opt_plan/03 section 4.2).
//
// The two arms differ ONLY in QwenEngineOptions::dsl_split_kv. Each generates 64
// greedy ids from a 504-token prompt, so the decode positions cross BOTH the
// 512-token sparse-KV page boundary AND the default split threshold (512) while
// the session runs -- the exact transition the statically guarded legacy/split/
// combine nodes exist for. Then the session is reset, re-prefilled and 8 more ids
// are generated, because a captured graph and the fixed partial buffers have to
// survive a new session.
//
// Runs on cuda (static graph: the counters must show one launch per step and ZERO
// updates, i.e. crossing the threshold needed no graph maintenance) and on dx/vk
// (command lists: the same ids must come out, with steps == 0).
// ---------------------------------------------------------------------------
struct SplitArm {
    luisa::vector<uint32_t> ids;
    luisa::vector<uint32_t> ids_after_reset;
    uint64_t steps{0};
    uint64_t updates{0};
    uint32_t position{0};
    bool split{false};
    bool dedup{false};
    uint32_t splits{0};
    uint32_t blocks{0};
    uint32_t threshold{0};
};

[[nodiscard]] int run_split_ab(Device &device, Stream &stream, const Qwen35Model &model,
                               const QwenDeviceWeights &weights, const char *backend) {
    const auto &g = model.config().geometry;
    const bool cuda = luisa::string_view{backend} == "cuda";
    luisa::vector<uint32_t> prompt(504u);
    uint64_t lcg = 987654321u;
    for (auto &id : prompt) {
        lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
        id = static_cast<uint32_t>((lcg >> 33u) % g.vocab);
    }
    constexpr uint32_t steps = 64u;
    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 1024u;
    options.prefill_chunk = 64u;
    options.dsl_decode_specialized = true;
    const auto generate = [&](QwenEngine &engine, uint32_t first_slot, uint32_t count,
                              uint32_t seed_token) {
        luisa::vector<uint32_t> ids;
        uint32_t pending = seed_token;
        bool need_host = true;
        for (uint32_t done = 0; done < count;) {
            const uint32_t chunk_len = std::min(4u, count - done);
            for (uint32_t j = 0; j < chunk_len; j++) {
                const auto status = engine.decode_device_greedy(
                    stream, need_host ? &pending : nullptr, first_slot + done + j);
                check(status.ok(), "split A/B: device greedy step");
                if (!status.ok()) { return ids; }
                need_host = false;
            }
            luisa::vector<uint32_t> chunk(chunk_len, UINT32_MAX);
            check(engine.download_device_greedy(stream, first_slot + done,
                                                luisa::span<uint32_t>{chunk}).ok(),
                  "split A/B: device greedy download");
            for (auto id : chunk) { ids.emplace_back(id); }
            done += chunk_len;
        }
        return ids;
    };
    const auto run_arm = [&](bool split_on) {
        SplitArm arm;
        options.dsl_split_kv = split_on;
        auto engine = QwenEngine::create(device, stream, model, weights, options);
        check(engine.ok(), split_on ? "split A/B: create split engine" : "split A/B: create legacy engine");
        if (!engine.ok()) { return arm; }
        luisa::vector<float> logits;
        check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits).ok(),
              "split A/B: prefill 504 tokens");
        auto next = qwen_argmax(logits);
        check(next.ok(), "split A/B: host argmax");
        if (!next.ok()) { return arm; }
        arm.ids = generate(**engine.value, 0u, steps, *next.value);
        {
            const auto info = (*engine.value)->info();
            arm.steps = info.cuda_graph_steps;
            arm.updates = info.cuda_graph_updates;
            arm.position = info.position;
            arm.split = info.decode_attn_split;
            arm.dedup = info.decode_attn_dedup;
            arm.splits = info.decode_attn_splits;
            arm.blocks = info.decode_attn_blocks;
            arm.threshold = info.decode_attn_threshold;
        }
        // Reset + re-prefill: the same engine, a fresh session. The split path must
        // produce the same ids again for the same prompt, i.e. neither the fixed
        // partial buffers nor a captured graph may carry state across a reset.
        check((*engine.value)->reset(stream).ok(), "split A/B: reset");
        check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits).ok(),
              "split A/B: re-prefill after reset");
        auto again = qwen_argmax(logits);
        check(again.ok(), "split A/B: host argmax after reset");
        if (!again.ok()) { return arm; }
        arm.ids_after_reset = generate(**engine.value, 0u, 8u, *again.value);
        std::printf("split A/B arm: split=%d steps=%llu updates=%llu position=%u ids=%s\n",
                    split_on ? 1 : 0,
                    static_cast<unsigned long long>(arm.steps),
                    static_cast<unsigned long long>(arm.updates), arm.position,
                    arm.ids.empty() ? "-" : "ok");
        return arm;
    };
    auto split_arm = run_arm(true);
    auto legacy_arm = run_arm(false);
    check(split_arm.ids.size() == steps && legacy_arm.ids.size() == steps,
          "split A/B: 64 greedy ids per arm");
    check(!split_arm.ids.empty() && split_arm.ids == legacy_arm.ids,
          "split A/B: split and legacy ids are identical (64 tokens across the split threshold and a KV page)");
    check(!split_arm.ids_after_reset.empty() &&
              split_arm.ids_after_reset == legacy_arm.ids_after_reset,
          "split A/B: ids after reset + re-prefill are identical");
    check(split_arm.split && !legacy_arm.split,
          "split A/B: the engine reports the realization it compiled");
    check(split_arm.threshold == 512u && split_arm.position == 504u + steps,
          "split A/B: default threshold and host position accounting");
    // The last step ran at kv_len == position (prepare appended at position-1), which
    // is past the threshold, so the engine must report the deduped path -- and the
    // legacy arm must not.
    check(split_arm.dedup && split_arm.splits > 0u &&
              split_arm.blocks > 16u,
          "split A/B: the split family served the final kv_len (dedup reported)");
    check(!legacy_arm.dedup && legacy_arm.splits == 0u && legacy_arm.blocks == 16u,
          "split A/B: the legacy arm reports the per-q_head kernel");
    if (cuda) {
        check(split_arm.steps == steps && split_arm.updates == 0u,
              "split A/B: one static-graph launch per step and zero updates across the threshold");
        check(legacy_arm.steps == steps && legacy_arm.updates == 0u,
              "split A/B: the legacy graph also needs no updates");
    } else {
        check(split_arm.steps == 0u && legacy_arm.steps == 0u,
              "split A/B: non-CUDA backends keep ordinary command lists");
    }
    return summarize("qwen split-KV A/B", backend);
}

}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend> [zero|fp8-zero|bf16-zero|graph-ab|split-ab]\n",
                     argv[0]);
        return 2;
    }
    const luisa::filesystem::path model_path = "data/Qwen3.5-4B-BF16.gguf";
    std::error_code ec;
    if (!luisa::filesystem::exists(model_path, ec)) {
        std::printf("SKIP: %s is not present\n", model_path.string().c_str());
        return 0;
    }
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    if (!device) { return 2; }
    auto stream = device.create_stream();
    auto reader = GgufReader::open(model_path);
    check(reader.ok(), "open real BF16 GGUF");
    if (!reader.ok()) { return summarize("qwen mtp executor", argv[1]); }
    auto model = Qwen35Model::load(**reader.value);
    check(model.ok(), "validate real trunk");
    if (!model.ok()) { return summarize("qwen mtp executor", argv[1]); }
    if (argc > 2 && (luisa::string_view{argv[2]} == "fp8-zero" ||
                     luisa::string_view{argv[2]} == "bf16-zero")) {
        if (luisa::string_view{argv[1]} != "cuda") {
            std::fprintf(stderr, "the static-graph zero probe requires cuda\n");
            return 2;
        }
        return run_zero_graph(device, stream, *model.value,
                              luisa::string_view{argv[2]} == "fp8-zero");
    }
    QwenWeightUploadOptions upload_options;
    const bool zero_mode = argc > 2 && luisa::string_view{argv[2]} == "zero";
    auto weights = zero_mode ? Result<QwenDeviceWeights>::success(make_zero_trunk(device, stream, *model.value))
                             : QwenDeviceWeights::upload(device, stream, **reader.value,
                                                        *model.value, upload_options);
    check(weights.ok(), zero_mode ? "create zero trunk" : "upload real trunk");
    if (!weights.ok()) { return summarize("qwen mtp executor", argv[1]); }
    if (argc > 2 && luisa::string_view{argv[2]} == "graph-ab") {
        if (luisa::string_view{argv[1]} != "cuda") {
            std::fprintf(stderr, "the static-graph A/B probe requires cuda\n");
            return 2;
        }
        return run_graph_ab(device, stream, *model.value, *weights.value);
    }

        if (argc > 2 && luisa::string_view{argv[2]} == "split-ab") {
            // Split-KV decode attention A/B on the real trunk (opt_plan/03 4.2).
            return run_split_ab(device, stream, *model.value, *weights.value, argv[1]);
        }
    auto draft_weights = make_zero_draft(device, stream, *model.value);
    const auto *embd = weights.value->find(QwenTensorRole::token_embd);
    check(embd != nullptr, "tied embedding present");
    check(QwenNextnDraft::validate_weights(*model.value, *embd, draft_weights).ok(),
          "synthetic draft passes strict inventory validation");

    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 32u;
    options.prefill_chunk = 8u;
    options.dsl_gdn_wy_ut = false;
    options.dsl_gdn_auto = false;// keep the plain/MTP numerical path identical// identical plain/MTP numerical path
    options.dsl_decode_specialized = false;// verify/replay/plain reference reduction order
    auto mtp = QwenMtpExecutor::create(device, stream, *model.value, *weights.value,
                                       draft_weights, options, 2u);
    check(mtp.ok(), "create MTP executor");
    auto plain = QwenEngine::create(device, stream, *model.value, *weights.value, options);
    check(plain.ok(), "create plain target");
    if (!mtp.ok() || !plain.ok()) { return summarize("qwen mtp executor", argv[1]); }

    luisa::vector<uint32_t> prompt;
    if (zero_mode) {
        // Cross two engine chunk boundaries and the draft's history re-prefill.
        prompt.assign(17u, 0u);
    } else {
        prompt = {9419u, 11u, 1204u, 513u, 488u, 30u};
    }
    luisa::vector<float> mtp_logits, plain_logits;
    check((*mtp.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, mtp_logits).ok(), "MTP prefill");
    check((*plain.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, plain_logits).ok(), "plain prefill");
    auto initial_mtp = qwen_argmax(mtp_logits);
    auto pending = qwen_argmax(plain_logits);
    check(initial_mtp.ok() && pending.ok() && *initial_mtp.value == *pending.value,
          "initial pending tokens match");
    if (!pending.ok()) { return summarize("qwen mtp executor", argv[1]); }

    const uint32_t round_limit = zero_mode ? 2u : 3u;
    for (uint32_t round_index = 0; round_index < round_limit && (*mtp.value)->can_speculate(); round_index++) {
        QwenMtpRoundResult round;
        auto status = (*mtp.value)->round(stream, *pending.value, QwenSampleOptions{}, round);
        check(status.ok(), "MTP draft+verify round");
        if (!status.ok()) { break; }
        check(!round.emitted.empty(), "round emits at least pending");
        if (zero_mode) {
            check(round.emitted.size() == 3u, "zero-model full accept emits pending + two drafts");
            check(round.accepted == 2u, "zero-model accepts every draft");
        }
        for (const auto id : round.emitted) {
            check(id == *pending.value, "speculative prefix == plain target");
            if (id != *pending.value) { break; }
            auto decode_status = (*plain.value)->decode(stream, *pending.value, plain_logits);
            check(decode_status.ok(), "plain target decode");
            if (!decode_status.ok()) { break; }
            pending = qwen_argmax(plain_logits);
            check(pending.ok(), "plain next pending argmax");
            if (!pending.ok()) { break; }
        }
        if (round.next_pending != *pending.value && pending.ok()) {
            float max_diff = 0.f;
            size_t max_i = 0u, exact = 0u;
            for (size_t i = 0; i < plain_logits.size(); i++) {
                const float d = std::abs(round.next_logits[i] - plain_logits[i]);
                if (d > max_diff) { max_diff = d; max_i = i; }
                exact += round.next_logits[i] == plain_logits[i] ? 1u : 0u;
            }
            std::fprintf(stderr,
                         "bonus mismatch round=%u mtp=%u plain=%u maxdiff=%g@%zu exact=%zu/%zu "
                         "mtp[ids]=%g,%g plain[ids]=%g,%g\n",
                         round_index, round.next_pending, *pending.value, max_diff, max_i,
                         exact, plain_logits.size(),
                         round.next_logits[round.next_pending], round.next_logits[*pending.value],
                         plain_logits[round.next_pending], plain_logits[*pending.value]);
        }
        check(round.next_pending == *pending.value, "bonus == plain next token");
        check((*mtp.value)->position() == (*plain.value)->info().position,
              "MTP/plain logical positions match after rollback");
        std::printf("round %u: emitted=%zu accepted=%u position=%u\n",
                    round_index, round.emitted.size(), round.accepted,
                    (*mtp.value)->position());
        pending = qwen_argmax(round.next_logits);
        check(pending.ok() && *pending.value == round.next_pending,
              "next logits produce pending bonus");
        if (!pending.ok()) { break; }
    }
    check((*mtp.value)->rounds() > 0u, "at least one real MTP round executed");
    check((*mtp.value)->drafted_total() > 0u, "actual NextN draft tokens produced");
    if (zero_mode) {
        check((*mtp.value)->accepted_total() == (*mtp.value)->drafted_total(),
              "zero-model full-accept statistics");
        check((*mtp.value)->position() == prompt.size() + 6u,
              "zero-model two rounds consume six emitted tokens");
    }
    std::printf("MTP stats: rounds=%llu accepted=%llu/%llu\n",
                static_cast<unsigned long long>((*mtp.value)->rounds()),
                static_cast<unsigned long long>((*mtp.value)->accepted_total()),
                static_cast<unsigned long long>((*mtp.value)->drafted_total()));

    if (zero_mode) {
        // Exact partial acceptance: target always chooses 0; proposal is
        // [accept 0, reject 1]. This exercises the a=1, two-row restore+replay.
        check((*mtp.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, mtp_logits).ok(),
              "partial-accept MTP prefill");
        check((*plain.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, plain_logits).ok(),
              "partial-accept plain prefill");
        pending = qwen_argmax(plain_logits);
        const uint32_t partial_drafts[] = {0u, 1u};
        QwenMtpRoundResult partial;
        auto partial_status = (*mtp.value)->verify(
            stream, *pending.value, luisa::span<const uint32_t>{partial_drafts},
            QwenSpeculativeLogits{}, QwenSampleOptions{}, partial);
        check(partial_status.ok(), "partial-accept external verify");
        check(partial.accepted == 1u && partial.emitted.size() == 2u,
              "partial accepts one of two drafts");
        for (const auto id : partial.emitted) {
            check(id == *pending.value, "partial replay prefix == plain");
            check((*plain.value)->decode(stream, *pending.value, plain_logits).ok(),
                  "partial plain decode");
            pending = qwen_argmax(plain_logits);
        }
        check(partial.next_pending == *pending.value, "partial bonus == plain next");
        check((*mtp.value)->position() == (*plain.value)->info().position,
              "partial multi-row replay position parity");
        check((*mtp.value)->position() == prompt.size() + 2u,
              "partial replay consumed exactly pending + accepted draft");
        check((*mtp.value)->accepted_total() == 1u && (*mtp.value)->drafted_total() == 2u,
              "partial acceptance statistics");

        // Sampling-mode verify is reproducible with identical seed/proposal.
        QwenSampleOptions sampling;
        sampling.temperature = 0.8f;
        sampling.seed = 12345u;
        const auto run_sampling = [&](QwenMtpRoundResult &result, uint32_t &pending_out) {
            luisa::vector<float> logits;
            auto status = (*mtp.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits);
            if (!status.ok()) { return status; }
            auto sampled = qwen_sample(logits, sampling, 0u);
            if (!sampled.ok()) { return sampled.status; }
            pending_out = *sampled.value;
            const uint32_t draft_ids[] = {1u, 2u};
            luisa::vector<float> zero_logits(2u * model.value->config().geometry.vocab, 0.0f);
            QwenSpeculativeLogits draft_view{zero_logits.data(), 2u,
                                             model.value->config().geometry.vocab,
                                             model.value->config().geometry.vocab};
            return (*mtp.value)->verify(stream, pending_out,
                                        luisa::span<const uint32_t>{draft_ids},
                                        draft_view, sampling, result);
        };
        QwenMtpRoundResult sample_a, sample_b;
        uint32_t sample_pending_a = 0u, sample_pending_b = 0u;
        check(run_sampling(sample_a, sample_pending_a).ok(), "sampling verify A");
        check(run_sampling(sample_b, sample_pending_b).ok(), "sampling verify B");
        check(sample_pending_a == sample_pending_b, "sampling pending reproducible");
        check(sample_a.emitted == sample_b.emitted && sample_a.next_pending == sample_b.next_pending,
              "sampling round reproducible");
        check(sample_a.accepted == 2u && sample_a.emitted.size() == 3u,
              "sampling equal-distribution all-accept path");
        check((*mtp.value)->position() == prompt.size() + 3u,
              "sampling round logical position");
    }
    stream.synchronize();
    return summarize("qwen mtp executor", argv[1]);
}
