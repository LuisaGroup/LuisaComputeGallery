// Qwen3.5-4B public engine interface -- factory dispatch and shared sampler.
// See engine.h for the contract. The factory creates EXACTLY the requested
// implementation (QwenDslEngine / QwenTileEngine); there is no DSL<->Tile
// fallback and no BF16<->FP8 substitution -- each engine validates its weight
// storage itself and fails closed on a mismatch.
#include "qwen/engine.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include "qwen/dsl_engine.h"
#include "qwen/tile_engine.h"

namespace luisa::qwen {

void QwenEngineDeleter::operator()(QwenEngine *p) const noexcept {
    if (p != nullptr) {
        std::destroy_at(p); // virtual destructor releases the derived engine
        luisa::deallocate_with_allocator(p);
    }
}

[[nodiscard]] luisa::string_view qwen_kernel_kind_name(QwenKernelKind kind) noexcept {
    switch (kind) {
        case QwenKernelKind::dsl: return "dsl";
        case QwenKernelKind::tile: return "tile";
    }
    return "unknown";
}

[[nodiscard]] Result<QwenEnginePtr> QwenEngine::create(
    compute::Device &device, compute::Stream &stream,
    const Qwen35Model &model, const QwenDeviceWeights &weights,
    const QwenEngineOptions &options) {
    using R = Result<QwenEnginePtr>;
    if (!device) {
        return R::failure(StatusCode::invalid_argument, "invalid device", "device");
    }
    if (!stream) {
        return R::failure(StatusCode::invalid_argument, "invalid stream", "stream");
    }
    switch (options.kernel) {
        case QwenKernelKind::dsl:
            return QwenDslEngine::create(device, stream, model, weights, options);
        case QwenKernelKind::tile:
            return QwenTileEngine::create(device, stream, model, weights, options);
    }
    return R::failure(StatusCode::invalid_argument,
                      "unknown QwenKernelKind (expected dsl or tile)", "options.kernel");
}

[[nodiscard]] Result<uint32_t> qwen_argmax(luisa::span<const float> logits) noexcept {
    using R = Result<uint32_t>;
    if (logits.empty()) {
        return R::failure(StatusCode::invalid_argument, "argmax over empty logits", "logits");
    }
    // Deterministic greedy selection: strictly-greater comparison keeps the
    // LOWEST index on ties. NaN anywhere is a recoverable failure (a NaN
    // logit means the forward pass produced garbage; silently skipping it
    // would hide the bug). Infinities are ordered normally.
    uint32_t best = 0;
    float best_value = logits[0];
    if (std::isnan(best_value)) {
        return R::failure(StatusCode::invalid_argument, "NaN logit at index 0", "logits");
    }
    for (uint32_t i = 1; i < static_cast<uint32_t>(logits.size()); i++) {
        const float v = logits[i];
        if (std::isnan(v)) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "NaN logit at index %u", i);
            return R::failure(StatusCode::invalid_argument, luisa::string{buf}, "logits");
        }
        if (v > best_value) {
            best_value = v;
            best = i;
        }
    }
    return R::success(best);
}

namespace {

[[nodiscard]] uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30u)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27u)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31u);
}

[[nodiscard]] double uniform_open01(uint64_t seed, uint64_t draw_index) noexcept {
    // Hash rather than advance a mutable RNG: draw_index is part of the random
    // identity. Taking 53 bits gives an exactly representable double in (0,1),
    // excluding both endpoints so categorical selection never reads past the
    // vocabulary and zero-probability entries are never selected spuriously.
    const auto bits = splitmix64(seed ^ splitmix64(draw_index)) >> 11u;
    return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
}

}// namespace

[[nodiscard]] Result<uint32_t> qwen_sample(
    luisa::span<const float> logits, const QwenSampleOptions &options,
    uint64_t draw_index) noexcept {
    using R = Result<uint32_t>;
    if (options.temperature == 0.0f) { return qwen_argmax(logits); }
    if (logits.empty()) {
        return R::failure(StatusCode::invalid_argument, "sampling over empty logits", "logits");
    }
    if (!(options.temperature > 0.0f) || !std::isfinite(options.temperature)) {
        return R::failure(StatusCode::invalid_argument,
                          "sampling temperature must be zero or finite and positive",
                          "options.temperature");
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(logits.size()); i++) {
        if (std::isnan(logits[i])) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "NaN logit at index %u", i);
            return R::failure(StatusCode::invalid_argument, luisa::string{buf}, "logits");
        }
    }
    const auto u = uniform_open01(options.seed, draw_index);

    // Infinite target logits have a well-defined limiting distribution: mass
    // is shared only among +infinity entries. Handle them before max/exp so
    // inf-inf never creates a NaN.
    uint32_t positive_inf_count = 0;
    for (const float v : logits) {
        if (v == std::numeric_limits<float>::infinity()) { positive_inf_count++; }
    }
    if (positive_inf_count != 0u) {
        auto rank = static_cast<uint32_t>(u * static_cast<double>(positive_inf_count));
        for (uint32_t i = 0; i < static_cast<uint32_t>(logits.size()); i++) {
            if (logits[i] == std::numeric_limits<float>::infinity() && rank-- == 0u) {
                return R::success(i);
            }
        }
        return R::failure(StatusCode::internal_error, "failed to select a +infinity logit", "logits");
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    for (const float v : logits) { max_logit = std::max(max_logit, static_cast<double>(v)); }
    if (!std::isfinite(max_logit)) {
        return R::failure(StatusCode::invalid_argument,
                          "categorical distribution has zero mass (all logits are -infinity)",
                          "logits");
    }
    double total = 0.0;
    const double inv_temperature = 1.0 / static_cast<double>(options.temperature);
    for (const float v : logits) {
        if (v != -std::numeric_limits<float>::infinity()) {
            total += std::exp((static_cast<double>(v) - max_logit) * inv_temperature);
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return R::failure(StatusCode::internal_error, "invalid categorical probability sum", "logits");
    }
    const double threshold = u * total;
    double cumulative = 0.0;
    uint32_t last_nonzero = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(logits.size()); i++) {
        const float v = logits[i];
        if (v == -std::numeric_limits<float>::infinity()) { continue; }
        const double weight = std::exp((static_cast<double>(v) - max_logit) * inv_temperature);
        if (weight > 0.0) { last_nonzero = i; }
        cumulative += weight;
        if (threshold < cumulative) { return R::success(i); }
    }
    // Guard only against the final floating-point accumulation boundary.
    return R::success(last_nonzero);
}

}// namespace luisa::qwen
