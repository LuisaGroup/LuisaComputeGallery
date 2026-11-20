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
[[nodiscard]] int run_fp8_zero_graph(Device &device, Stream &stream, const Qwen35Model &model) {
    auto weights = make_zero_trunk(device, stream, model, true);
    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 32u;
    options.prefill_chunk = 8u;
    options.dsl_decode_specialized = true;
    options.dsl_cuda_graph = true;
    auto engine = QwenEngine::create(device, stream, model, weights, options);
    check(engine.ok(), "create FP8 zero engine");
    if (!engine.ok()) { return summarize("qwen FP8 graph probe", "cuda"); }
    const luisa::vector<uint32_t> prompt(8u, 0u);
    luisa::vector<float> logits;
    check((*engine.value)->prefill(stream, luisa::span<const uint32_t>{prompt}, logits).ok(),
          "FP8 zero prefill");
    uint32_t token = 0u;
    for (uint32_t step = 0; step < 4u; ++step) {
        const auto status = (*engine.value)->decode_device_greedy(
            stream, step == 0u ? &token : nullptr, step);
        check(status.ok(), "FP8 zero graph decode step");
        if (!status.ok()) { return summarize("qwen FP8 graph probe", "cuda"); }
    }
    luisa::vector<uint32_t> history(4u, UINT32_MAX);
    check((*engine.value)->download_device_greedy(stream, 0u, luisa::span<uint32_t>{history}).ok(),
          "FP8 zero graph history download");
    check(history == luisa::vector<uint32_t>(4u, 0u), "FP8 zero graph deterministic history");
    const auto info = (*engine.value)->info();
    check(info.cuda_graph_steps == 4u, "FP8 graph launch counter");
    check(info.cuda_graph_updates == 3u, "FP8 graph update counter");
    return summarize("qwen FP8 graph probe", "cuda");
}

}// namespace

int main(int argc, char **argv) {
    unbuffered_output();
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <backend> [zero|fp8-zero]\n", argv[0]);
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
    if (argc > 2 && luisa::string_view{argv[2]} == "fp8-zero") {
        if (luisa::string_view{argv[1]} != "cuda") {
            std::fprintf(stderr, "fp8-zero graph probe requires cuda\n");
            return 2;
        }
        return run_fp8_zero_graph(device, stream, *model.value);
    }
    QwenWeightUploadOptions upload_options;
    const bool zero_mode = argc > 2 && luisa::string_view{argv[2]} == "zero";
    auto weights = zero_mode ? Result<QwenDeviceWeights>::success(make_zero_trunk(device, stream, *model.value))
                             : QwenDeviceWeights::upload(device, stream, **reader.value,
                                                        *model.value, upload_options);
    check(weights.ok(), zero_mode ? "create zero trunk" : "upload real trunk");
    if (!weights.ok()) { return summarize("qwen mtp executor", argv[1]); }
    auto draft_weights = make_zero_draft(device, stream, *model.value);
    const auto *embd = weights.value->find(QwenTensorRole::token_embd);
    check(embd != nullptr, "tied embedding present");
    check(QwenNextnDraft::validate_weights(*model.value, *embd, draft_weights).ok(),
          "synthetic draft passes strict inventory validation");

    QwenEngineOptions options;
    options.kernel = QwenKernelKind::dsl;
    options.max_context = 32u;
    options.prefill_chunk = 8u;
    options.dsl_gdn_wy_ut = false;// identical plain/MTP numerical path
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
