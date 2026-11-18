// Qwen3.5-4B shared engine layer -- isolated host-only MTP eligibility /
// default-policy probe. See qwen/mtp.h for the honesty contract.
// Fail-closed: any malformed/incomplete inventory yields available=false
// with a precise reason; nothing here can enable MTP by default.
#include "qwen/mtp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>

#include "qwen/model.h"

namespace luisa::qwen {
namespace {

// One canonical draft tensor expectation.
struct MtpTensorSpec {
    const char *name;
    bool required;
    bool dense_f32;// sensitive tensor: must be stored F32 in the GGUF
    uint64_t d0;
    uint64_t d1;// 0 => rank-1 tensor
};

// Fixed order: 15 required tensors, then 2 optional shared-head tensors.
// Shapes follow the frozen Qwen3.5-4B geometry (see mtp.h and model.h) and
// llama.cpp's load_block_mtp; dtype rules follow the qwen weight-storage
// contract (norms stay dense F32; matrices BF16/F32, Q8_0 detected but not
// usable by the BF16/FP8 engine paths without explicit dequantization).
[[nodiscard]] const MtpTensorSpec *mtp_tensor_specs(uint64_t &count) noexcept {
    static constexpr MtpTensorSpec specs[] = {
        {"blk.32.eh_proj.weight", true, false, 5120, 2560},
        {"blk.32.enorm.weight", true, true, 2560, 0},
        {"blk.32.hnorm.weight", true, true, 2560, 0},
        {"blk.32.shared_head.norm.weight", true, true, 2560, 0},
        {"blk.32.attn_norm.weight", true, true, 2560, 0},
        {"blk.32.post_attention_norm.weight", true, true, 2560, 0},
        {"blk.32.attn_q.weight", true, false, 2560, 8192},
        {"blk.32.attn_k.weight", true, false, 2560, 1024},
        {"blk.32.attn_v.weight", true, false, 2560, 1024},
        {"blk.32.attn_output.weight", true, false, 4096, 2560},
        {"blk.32.attn_q_norm.weight", true, true, 256, 0},
        {"blk.32.attn_k_norm.weight", true, true, 256, 0},
        {"blk.32.ffn_gate.weight", true, false, 2560, 9216},
        {"blk.32.ffn_up.weight", true, false, 2560, 9216},
        {"blk.32.ffn_down.weight", true, false, 9216, 2560},
        // Optional: only present when mtp_use_dedicated_embeddings=true or
        // an untied dedicated LM head was exported. Qwen3.5-4B shares the
        // trunk embedding / tied head, so absence is normal.
        {"blk.32.token_embd.weight", false, false, 2560, 248320},
        {"blk.32.output.weight", false, false, 2560, 248320},
    };
    count = sizeof(specs) / sizeof(specs[0]);
    return specs;
}

[[nodiscard]] luisa::string dims_string(luisa::span<const uint64_t> dims) {
    luisa::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) { s += ", "; }
        s += luisa::format("{}", dims[i]);
    }
    s += "]";
    return s;
}

// The current checkpoint's situation: no nextn metadata at all AND no
// blk.32.* tensor at all. Reported as plain "missing weights".
[[nodiscard]] bool is_pristine_missing_weights(const MtpEligibilityReport &r) noexcept {
    return !r.nextn_metadata_present && r.detected_tensor_count == 0;
}

}// namespace

[[nodiscard]] luisa::string_view mtp_tensor_state_name(MtpTensorState s) noexcept {
    switch (s) {
        case MtpTensorState::missing: return "missing";
        case MtpTensorState::malformed: return "malformed";
        case MtpTensorState::ok: return "ok";
    }
    return "unknown";
}

[[nodiscard]] MtpEligibilityReport probe_mtp_eligibility(const GgufReader &reader) {
    MtpEligibilityReport r;

    // ---- metadata evidence --------------------------------------------------
    if (const auto arch = reader.meta_string("general.architecture"); arch.has_value()) {
        r.architecture = luisa::string{*arch};
    }
    r.architecture_is_qwen35 = r.architecture == "qwen35";
    if (const auto *kv = reader.metadata(kMtpNextnMetadataKey); kv != nullptr) {
        r.nextn_metadata_present = true;
        if (const auto n = reader.meta_u64(kMtpNextnMetadataKey); n.has_value()) {
            r.nextn_predict_layers = *n;
            r.nextn_metadata_valid = true;// range checked against block_count below
        }
    }
    if (r.architecture_is_qwen35) {
        r.block_count = reader.meta_u64("qwen35.block_count");
    }
    if (r.nextn_metadata_valid && r.block_count.has_value() &&
        r.nextn_predict_layers > *r.block_count) {
        r.nextn_metadata_valid = false;
    }

    // ---- inventory evidence --------------------------------------------------
    r.total_tensor_count = reader.tensors().size();
    constexpr luisa::string_view kDraftPrefix = "blk.32.";
    for (const auto &t : reader.tensors()) {
        if (luisa::string_view{t.name}.starts_with(kDraftPrefix)) {
            r.detected_tensor_names.emplace_back(t.name);
        }
    }
    std::sort(r.detected_tensor_names.begin(), r.detected_tensor_names.end());
    r.detected_tensor_count = r.detected_tensor_names.size();

    uint64_t spec_count = 0;
    const auto *specs = mtp_tensor_specs(spec_count);
    for (uint64_t i = 0; i < spec_count; ++i) {
        const auto &spec = specs[i];
        MtpTensorCheck c;
        c.name = spec.name;
        c.required = spec.required;
        c.expected_dims.emplace_back(spec.d0);
        if (spec.d1 != 0) { c.expected_dims.emplace_back(spec.d1); }
        const auto *info = reader.find(spec.name);
        if (info == nullptr) {
            c.state = MtpTensorState::missing;
            if (spec.required) { ++r.required_missing_count; }
        } else {
            if (!spec.required) { ++r.optional_present_count; }
            c.actual_dims = info->dims;
            c.actual_type = info->type;
            const bool shape_ok =
                info->dims.size() == c.expected_dims.size() &&
                std::equal(info->dims.begin(), info->dims.end(), c.expected_dims.begin());
            bool dtype_ok = false;
            if (spec.dense_f32) {
                dtype_ok = info->type == static_cast<uint32_t>(GgufTensorType::f32);
                if (!dtype_ok) {
                    c.issue = luisa::format("sensitive tensor must be stored F32 (found {})",
                                            gguf_tensor_type_name(info->type));
                }
            } else if (info->type == static_cast<uint32_t>(GgufTensorType::q8_0)) {
                // Q8_0 is INT8 block quantization, never FP8; the engine's
                // BF16/FP8 weight paths cannot consume it without explicit
                // dequantization. Fail closed for the draft contract.
                c.issue = "Q8_0 draft matrix is not supported by the BF16/FP8 engine paths "
                          "(explicit dequantization would be required)";
            } else {
                dtype_ok = gguf_tensor_type_supported(info->type);
                if (!dtype_ok) {
                    c.issue = luisa::format("unsupported tensor dtype {}",
                                            gguf_tensor_type_name(info->type));
                }
            }
            if (!shape_ok) {
                if (!c.issue.empty()) { c.issue += "; "; }
                c.issue += luisa::format("shape {} does not match the frozen geometry {}",
                                         dims_string(info->dims), dims_string(c.expected_dims));
            }
            c.state = (shape_ok && dtype_ok) ? MtpTensorState::ok : MtpTensorState::malformed;
            if (spec.required) {
                if (c.state == MtpTensorState::ok) {
                    ++r.required_ok_count;
                } else {
                    ++r.required_malformed_count;
                }
            } else if (c.state == MtpTensorState::malformed) {
                // Optional means "may be absent", never "may be broken".
                ++r.optional_malformed_count;
            }
        }
        r.tensor_checks.emplace_back(std::move(c));
    }
    // Any blk.32.* tensor outside the canonical set (GDN/ssm tensors from a
    // wrong layer type, quant scales, renamed variants, ...) is unexpected.
    for (const auto &name : r.detected_tensor_names) {
        bool known = false;
        for (uint64_t i = 0; i < spec_count; ++i) {
            if (name == specs[i].name) {
                known = true;
                break;
            }
        }
        if (!known) { r.unexpected_tensor_names.emplace_back(name); }
    }

    // ---- verdicts -------------------------------------------------------------
    const bool metadata_one_layer =
        r.nextn_metadata_present && r.nextn_metadata_valid && r.nextn_predict_layers == 1u;
    const bool block_count_ok =
        r.block_count.has_value() &&
        *r.block_count == kMtpTrunkBlockCount + (metadata_one_layer ? 1u : 0u);
    r.draft_complete =
        r.architecture_is_qwen35 && metadata_one_layer && block_count_ok &&
        r.required_ok_count == kMtpRequiredTensorCount &&
        r.optional_malformed_count == 0 &&
        r.unexpected_tensor_names.empty();
    r.available = r.draft_complete;
    // Report trunk validity independently of draft availability. Combined
    // files require the opt-in NextN-tolerant trunk check; the ordinary engine
    // loader intentionally remains exact-426. Benchmark feasibility needs both.
    r.target_trunk_ok = r.draft_complete ? Qwen35Model::load_with_nextn(reader).ok()
                                         : Qwen35Model::load(reader).ok();
    r.benchmark_feasible = r.draft_complete && r.target_trunk_ok;

    // ---- default policy: never optimistic --------------------------------------
    // This module performs NO benchmark. benchmarked / benchmark_passed /
    // accelerates / default_enabled stay false unconditionally; only a real
    // measured greedy draft+verify A/B with full state-rollback correctness
    // and a net decode speedup may ever flip the default (see mtp.h).
    r.benchmarked = false;
    r.benchmark_passed = false;
    r.accelerates = false;
    r.default_enabled = false;

    // ---- reason ------------------------------------------------------------------
    if (!r.architecture_is_qwen35) {
        r.reason = luisa::format(
            "MTP unavailable: GGUF architecture is '{}' (expected 'qwen35'); "
            "the frozen Qwen3.5-4B NextN contract does not apply. MTP stays disabled.",
            r.architecture.empty() ? "<missing>" : r.architecture);
    } else if (is_pristine_missing_weights(r)) {
        r.reason = luisa::format(
            "MTP draft weights are missing from this GGUF: no '{}' metadata key and no "
            "blk.32.* tensors among the {} tensors. The HF config declares 1 MTP layer "
            "(mtp_num_hidden_layers=1) and the official HF checkpoint carries mtp.* weights, "
            "but this export omitted them. available=false, benchmarked=false, "
            "accelerates=false; MTP stays disabled by default.",
            kMtpNextnMetadataKey, r.total_tensor_count);
    } else if (!r.nextn_metadata_present) {
        r.reason = luisa::format(
            "MTP fails closed: {} blk.32.* tensor(s) detected but the '{}' metadata key is "
            "absent, so the NextN block count is undeclared. MTP stays disabled.",
            r.detected_tensor_count, kMtpNextnMetadataKey);
    } else if (!r.nextn_metadata_valid) {
        r.reason = luisa::format(
            "MTP fails closed: '{}' is present but not a valid integer count within "
            "qwen35.block_count. MTP stays disabled.",
            kMtpNextnMetadataKey);
    } else if (r.nextn_predict_layers != 1u) {
        r.reason = luisa::format(
            "MTP fails closed: '{}' = {} but this engine only contracts the one-layer "
            "Qwen3.5-4B draft (blk.32). MTP stays disabled.",
            kMtpNextnMetadataKey, r.nextn_predict_layers);
    } else if (!block_count_ok) {
        r.reason = luisa::format(
            "MTP fails closed: qwen35.block_count = {} does not equal 33 (32 trunk blocks + "
            "1 NextN draft block) required by the one-layer draft contract. MTP stays disabled.",
            r.block_count.has_value() ? luisa::format("{}", *r.block_count) : luisa::string{"<missing>"});
    } else if (!r.unexpected_tensor_names.empty()) {
        r.reason = luisa::format(
            "MTP fails closed: {} unexpected blk.32.* tensor(s) outside the canonical "
            "full-attention NextN set (first: {}). MTP stays disabled.",
            r.unexpected_tensor_names.size(), r.unexpected_tensor_names.front());
    } else if (r.optional_malformed_count != 0) {
        luisa::string first_issue;
        for (const auto &c : r.tensor_checks) {
            if (!c.required && c.state == MtpTensorState::malformed) {
                first_issue = luisa::format("{} malformed ({})", c.name, c.issue);
                break;
            }
        }
        r.reason = luisa::format(
            "MTP fails closed: {} optional blk.32 tensor(s) present but malformed -- optional "
            "means may-be-absent, never may-be-broken ({}). MTP stays disabled.",
            r.optional_malformed_count, first_issue);
    } else if (r.required_missing_count != 0 || r.required_malformed_count != 0) {
        luisa::string first_issue;
        for (const auto &c : r.tensor_checks) {
            if (c.required && c.state != MtpTensorState::ok) {
                first_issue = c.state == MtpTensorState::missing
                                  ? luisa::format("{} missing", c.name)
                                  : luisa::format("{} malformed ({})", c.name, c.issue);
                break;
            }
        }
        r.reason = luisa::format(
            "MTP fails closed: incomplete/malformed draft inventory -- {} of {} required "
            "tensors ok, {} missing, {} malformed ({}). MTP stays disabled.",
            r.required_ok_count, kMtpRequiredTensorCount, r.required_missing_count,
            r.required_malformed_count, first_issue);
    } else if (!r.target_trunk_ok) {
        r.reason = "MTP draft block is complete, but the target trunk in this GGUF does not "
                   "validate against the frozen Qwen3.5-4B inventory, so no honest A/B "
                   "target-vs-draft benchmark can run. MTP stays disabled by default.";
    } else {
        r.reason = "Complete one-layer Qwen3.5 NextN draft detected and the target trunk "
                   "validates: an A/B benchmark CAN run, but none has been measured. "
                   "accelerates is unproven; default_enabled stays false until a real greedy "
                   "draft+verify A/B passes with exact output equivalence, full state rollback "
                   "(KV logical length + GDN recurrent/conv state + position/mRoPE delta) and "
                   "a net measured decode speedup.";
    }
    return r;
}

[[nodiscard]] luisa::vector<luisa::string> MtpEligibilityReport::report_lines() const {
    luisa::vector<luisa::string> lines;
    lines.emplace_back(luisa::format(
        "mtp: architecture='{}' nextn_metadata={} nextn_predict_layers={} block_count={}",
        architecture.empty() ? "<missing>" : architecture,
        nextn_metadata_present ? (nextn_metadata_valid ? "present" : "present-invalid") : "absent",
        nextn_predict_layers,
        block_count.has_value() ? luisa::format("{}", *block_count) : luisa::string{"<missing>"}));
    lines.emplace_back(luisa::format(
        "mtp: tensors total={} detected_blk32={} required ok/missing/malformed={}/{}/{} "
        "optional_present={} optional_malformed={} unexpected={}",
        total_tensor_count, detected_tensor_count, required_ok_count, required_missing_count,
        required_malformed_count, optional_present_count, optional_malformed_count,
        unexpected_tensor_names.size()));
    for (const auto &c : tensor_checks) {
        if (c.state == MtpTensorState::ok) {
            lines.emplace_back(luisa::format("mtp:   {} ok ({}, {})", c.name,
                                             dims_string(c.actual_dims),
                                             gguf_tensor_type_name(c.actual_type)));
        } else if (c.state == MtpTensorState::missing) {
            lines.emplace_back(luisa::format("mtp:   {} missing (expected {}{})", c.name,
                                             dims_string(c.expected_dims),
                                             c.required ? "" : ", optional"));
        } else {
            lines.emplace_back(luisa::format("mtp:   {} MALFORMED: {}", c.name, c.issue));
        }
    }
    for (const auto &name : unexpected_tensor_names) {
        lines.emplace_back(luisa::format("mtp:   {} UNEXPECTED (outside canonical NextN set)", name));
    }
    lines.emplace_back(luisa::format(
        "mtp: draft_complete={} target_trunk_ok={} available={} benchmark_feasible={}",
        draft_complete, target_trunk_ok, available, benchmark_feasible));
    lines.emplace_back(luisa::format(
        "mtp: benchmarked={} benchmark_passed={} accelerates={} default_enabled={}",
        benchmarked, benchmark_passed, accelerates, default_enabled));
    lines.emplace_back("mtp: reason: " + reason);
    return lines;
}

// ---------------------------------------------------------------------------
// Device-free self test.
// ---------------------------------------------------------------------------
namespace {

// Minimal GGUF v3 byte builder (same wire format the real reader parses).
struct MtpFixtureBuilder {
    luisa::vector<std::byte> bytes;

    void u32(uint32_t x) {
        for (uint32_t i = 0; i < 4u; ++i) {
            bytes.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu));
        }
    }
    void u64(uint64_t x) {
        for (uint32_t i = 0; i < 8u; ++i) {
            bytes.emplace_back(static_cast<std::byte>((x >> (8u * i)) & 0xffu));
        }
    }
    void str(luisa::string_view s) {
        u64(s.size());
        for (auto c : s) { bytes.emplace_back(static_cast<std::byte>(c)); }
    }
    void pad32() {
        while (bytes.size() % 32u != 0u) { bytes.emplace_back(std::byte{0}); }
    }
};

struct MtpFixtureTensor {
    luisa::string name;
    uint32_t type;
    luisa::vector<uint64_t> dims;
};

// Stored payload size of one fixture tensor (F32=4 / BF16=2 bytes per
// element; Q8_0 = 34 bytes per 32-element block).
[[nodiscard]] uint64_t mtp_fixture_stored_size(const MtpFixtureTensor &t) noexcept {
    uint64_t n = 1;
    for (auto d : t.dims) { n *= d; }
    if (t.type == static_cast<uint32_t>(GgufTensorType::q8_0)) {
        return (n / 32u) * 34u;// 2-byte scale + 32 int8 codes per block
    }
    const uint64_t per = t.type == static_cast<uint32_t>(GgufTensorType::bf16) ? 2u : 4u;
    return n * per;
}

// Build a small synthetic GGUF: metadata (key -> pre-encoded value blob with
// the 4-byte type tag first) + tensors with real zero payloads sized exactly
// to the declared geometry (the reader validates declared sizes, bounds and
// non-overlap against the real file size). The probe never reads payloads.
[[nodiscard]] luisa::vector<std::byte> build_mtp_fixture(
    luisa::span<const std::pair<luisa::string, luisa::vector<std::byte>>> metas,
    luisa::span<const MtpFixtureTensor> tensors) {
    MtpFixtureBuilder b;
    const auto header = [&](luisa::span<const uint64_t> offsets) {
        b.bytes.clear();
        b.u32(0x46554747u);// "GGUF"
        b.u32(3u);         // version
        b.u64(tensors.size());
        b.u64(metas.size());
        for (const auto &[key, value] : metas) {
            b.str(key);
            b.bytes.insert(b.bytes.end(), value.begin(), value.end());
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto &t = tensors[i];
            b.str(t.name);
            b.u32(static_cast<uint32_t>(t.dims.size()));
            for (auto d : t.dims) { b.u64(d); }
            b.u32(t.type);
            b.u64(offsets.empty() ? 0 : offsets[i]);
        }
        b.pad32();
    };
    const auto stored_size = [](const MtpFixtureTensor &t) -> uint64_t {
        return mtp_fixture_stored_size(t);
    };
    luisa::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto &t : tensors) {
        offsets.emplace_back(rel);
        rel += stored_size(t);
        while (rel % 32u != 0u) { ++rel; }
    }
    header({});
    const uint64_t data_start = b.bytes.size();
    header(offsets);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const uint64_t abs = data_start + offsets[i];
        while (b.bytes.size() < abs) { b.bytes.emplace_back(std::byte{0}); }
        const uint64_t n = stored_size(tensors[i]);
        for (uint64_t k = 0; k < n; ++k) { b.bytes.emplace_back(std::byte{0}); }
    }
    return std::move(b.bytes);
}

[[nodiscard]] luisa::vector<std::byte> meta_u32_blob(uint32_t v) {
    MtpFixtureBuilder b;
    b.u32(4u);// GgufMetaType::u32
    b.u32(v);
    return std::move(b.bytes);
}
[[nodiscard]] luisa::vector<std::byte> meta_string_blob(luisa::string_view s) {
    MtpFixtureBuilder b;
    b.u32(8u);// GgufMetaType::string
    b.str(s);
    return std::move(b.bytes);
}
[[nodiscard]] luisa::vector<std::byte> meta_f32_blob(float v) {
    MtpFixtureBuilder b;
    b.u32(6u);// GgufMetaType::f32
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    b.u32(bits);
    return std::move(b.bytes);
}
[[nodiscard]] luisa::vector<std::byte> meta_u32_array_blob(luisa::span<const uint32_t> values) {
    MtpFixtureBuilder b;
    b.u32(9u);// GgufMetaType::array
    b.u32(4u);// element type u32
    b.u64(values.size());
    for (auto v : values) { b.u32(v); }
    return std::move(b.bytes);
}

struct MtpSelfTestCtx {
    luisa::vector<luisa::string> *report;
    uint64_t checks{0};
    uint64_t failures{0};

    void note(luisa::string line) const {
        if (report != nullptr) { report->emplace_back(std::move(line)); }
    }
    void expect(bool cond, luisa::string what) {
        ++checks;
        if (!cond) {
            ++failures;
            note("FAIL: " + what);
        }
    }
};

// Open a writable fixture file; false only on filesystem failure.
[[nodiscard]] std::FILE *open_fixture_write(MtpSelfTestCtx &ctx, const luisa::filesystem::path &dir,
                                            luisa::string_view tag,
                                            luisa::filesystem::path &path_out) {
    path_out = dir / (luisa::string{tag} + ".gguf");
    std::FILE *f = nullptr;
#ifdef _WIN32
    f = _wfopen(path_out.c_str(), L"wb");
#else
    f = std::fopen(path_out.c_str(), "wb");
#endif
    if (f == nullptr) {
        ctx.expect(false, luisa::format("cannot write fixture {}", tag));
    }
    return f;
}

// Write fixture bytes; false only on filesystem failure.
[[nodiscard]] bool write_fixture_file(MtpSelfTestCtx &ctx, const luisa::filesystem::path &dir,
                                      luisa::string_view tag, luisa::span<const std::byte> bytes,
                                      luisa::filesystem::path &path_out) {
    auto *f = open_fixture_write(ctx, dir, tag, path_out);
    if (f == nullptr) { return false; }
    const auto written = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (written != bytes.size()) {
        ctx.expect(false, luisa::format("short write for fixture {}", tag));
        return false;
    }
    return true;
}

// Stream a FULL-inventory fixture (real frozen shapes => multi-GB of zero
// payload, e.g. token_embd [2560, 248320]). The REAL GgufReader validates
// every declared payload size against the real file size, so the payload
// must genuinely exist -- but it is all zeros and neither the probe nor the
// model/config validators ever read it, so it is streamed in 1 MiB chunks
// (never buffered in host memory).
[[nodiscard]] bool write_streamed_fixture(MtpSelfTestCtx &ctx, const luisa::filesystem::path &dir,
                                          luisa::string_view tag,
                                          luisa::span<const std::pair<luisa::string, luisa::vector<std::byte>>> metas,
                                          luisa::span<const MtpFixtureTensor> tensors,
                                          luisa::filesystem::path &path_out) {
    // Header with placeholder offsets, then rebuild with the real ones (the
    // header length is offset-independent, so data_start stays identical).
    MtpFixtureBuilder b;
    const auto header = [&](luisa::span<const uint64_t> offsets) {
        b.bytes.clear();
        b.u32(0x46554747u);// "GGUF"
        b.u32(3u);         // version
        b.u64(tensors.size());
        b.u64(metas.size());
        for (const auto &[key, value] : metas) {
            b.str(key);
            b.bytes.insert(b.bytes.end(), value.begin(), value.end());
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto &t = tensors[i];
            b.str(t.name);
            b.u32(static_cast<uint32_t>(t.dims.size()));
            for (auto d : t.dims) { b.u64(d); }
            b.u32(t.type);
            b.u64(offsets.empty() ? 0 : offsets[i]);
        }
        b.pad32();
    };
    luisa::vector<uint64_t> offsets;
    uint64_t rel = 0;
    for (const auto &t : tensors) {
        offsets.emplace_back(rel);
        rel += mtp_fixture_stored_size(t);
        while (rel % 32u != 0u) { ++rel; }
    }
    header({});
    header(offsets);
    auto *f = open_fixture_write(ctx, dir, tag, path_out);
    if (f == nullptr) { return false; }
    bool ok = std::fwrite(b.bytes.data(), 1, b.bytes.size(), f) == b.bytes.size();
    static constexpr size_t kChunk = 1u << 20u;
    const luisa::vector<std::byte> zeros(kChunk, std::byte{0});
    uint64_t pos = b.bytes.size();
    const uint64_t data_start = pos;
    for (size_t i = 0; ok && i < tensors.size(); ++i) {
        const uint64_t abs = data_start + offsets[i];
        while (ok && pos < abs) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunk, abs - pos));
            ok = std::fwrite(zeros.data(), 1, n, f) == n;
            pos += n;
        }
        uint64_t remaining = mtp_fixture_stored_size(tensors[i]);
        while (ok && remaining != 0) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunk, remaining));
            ok = std::fwrite(zeros.data(), 1, n, f) == n;
            remaining -= n;
            pos += n;
        }
    }
    std::fclose(f);
    if (!ok) {
        ctx.expect(false, luisa::format("short write for streamed fixture {}", tag));
        return false;
    }
    return true;
}

// Open a fixture through the REAL GgufReader and probe it. `open_rejected`
// (optional) is set when the reader itself refuses the file (e.g. an
// unsupported tensor dtype): that is an even earlier fail-closed outcome.
[[nodiscard]] bool probe_fixture(MtpSelfTestCtx &ctx, const luisa::filesystem::path &dir,
                                 luisa::string_view tag,
                                 luisa::span<const std::pair<luisa::string, luisa::vector<std::byte>>> metas,
                                 luisa::span<const MtpFixtureTensor> tensors,
                                 MtpEligibilityReport &out, bool *open_rejected = nullptr) {
    const auto bytes = build_mtp_fixture(metas, tensors);
    luisa::filesystem::path path;
    if (!write_fixture_file(ctx, dir, tag, bytes, path)) { return false; }
    auto reader = GgufReader::open(path);
    if (!reader.ok()) {
        if (open_rejected != nullptr) {
            *open_rejected = true;
            return false;
        }
        ctx.expect(false, luisa::format("fixture {} failed to open: {}", tag, reader.status.message));
        return false;
    }
    out = probe_mtp_eligibility(**reader.value);
    return true;
}

// Canonical draft tensor list for fixtures (all ok).
[[nodiscard]] luisa::vector<MtpFixtureTensor> canonical_draft_tensors() {
    uint64_t count = 0;
    const auto *specs = mtp_tensor_specs(count);
    luisa::vector<MtpFixtureTensor> out;
    for (uint64_t i = 0; i < count; ++i) {
        const auto &s = specs[i];
        if (!s.required) { continue; }
        MtpFixtureTensor t;
        t.name = s.name;
        t.type = static_cast<uint32_t>(s.dense_f32 ? GgufTensorType::f32 : GgufTensorType::bf16);
        t.dims.emplace_back(s.d0);
        if (s.d1 != 0) { t.dims.emplace_back(s.d1); }
        out.emplace_back(std::move(t));
    }
    return out;
}

// The FULL valid frozen-geometry trunk tensor set (all 426 tensors, exact
// canonical names/shapes/dtypes). Written via write_streamed_fixture with
// real-size zero payloads (~8.7 GB, streamed): the REAL GgufReader demands
// genuinely sized payloads, while neither the probe nor Qwen35Model::load
// ever reads payload bytes. This is a complete, valid trunk inventory --
// NOT a draft-only fake positive.
[[nodiscard]] luisa::vector<MtpFixtureTensor> full_trunk_tensors() {
    const Qwen35Geometry g{};
    using R = QwenTensorRole;
    const R shared[] = {R::attn_norm, R::post_attention_norm, R::ffn_gate, R::ffn_up, R::ffn_down};
    const R attention[] = {R::attn_q, R::attn_k, R::attn_v, R::attn_output, R::attn_q_norm, R::attn_k_norm};
    const R gdn[] = {R::gdn_qkv, R::gdn_gate, R::gdn_alpha, R::gdn_beta, R::gdn_out,
                     R::gdn_conv1d, R::gdn_a, R::gdn_dt_bias, R::gdn_norm};
    const auto dims_of = [&](R role) -> luisa::vector<uint64_t> {
        switch (role) {
            case R::token_embd: return {g.hidden, g.vocab};
            case R::output_norm:
            case R::attn_norm:
            case R::post_attention_norm: return {g.hidden};
            case R::ffn_gate:
            case R::ffn_up: return {g.hidden, g.ffn};
            case R::ffn_down: return {g.ffn, g.hidden};
            case R::attn_q: return {g.hidden, g.q_proj_n};
            case R::attn_k:
            case R::attn_v: return {g.hidden, g.kv_proj_n};
            case R::attn_output: return {g.o_proj_k, g.hidden};
            case R::attn_q_norm:
            case R::attn_k_norm: return {g.head_dim};
            case R::gdn_qkv: return {g.hidden, g.gdn_qkv_n};
            case R::gdn_gate: return {g.hidden, g.gdn_z_n};
            case R::gdn_alpha:
            case R::gdn_beta: return {g.hidden, g.gdn_gate_n};
            case R::gdn_out: return {g.gdn_out_k, g.hidden};
            case R::gdn_conv1d: return {g.conv_kernel, g.gdn_qkv_n};
            case R::gdn_a:
            case R::gdn_dt_bias: return {g.gdn_v_heads};
            case R::gdn_norm: return {g.gdn_head_dim};
        }
        return {};
    };
    const auto f32_required = [](R role) noexcept {
        switch (role) {
            case R::output_norm:
            case R::attn_norm:
            case R::post_attention_norm:
            case R::attn_q_norm:
            case R::attn_k_norm:
            case R::gdn_conv1d:
            case R::gdn_a:
            case R::gdn_dt_bias:
            case R::gdn_norm: return true;
            default: return false;
        }
    };
    luisa::vector<MtpFixtureTensor> out;
    const auto add = [&](R role, uint64_t layer) {
        MtpFixtureTensor t;
        t.name = qwen_tensor_name(role, layer);
        t.type = static_cast<uint32_t>(f32_required(role) ? GgufTensorType::f32 : GgufTensorType::bf16);
        t.dims = dims_of(role);
        out.emplace_back(std::move(t));
    };
    add(R::token_embd, UINT64_MAX);
    add(R::output_norm, UINT64_MAX);
    for (uint64_t layer = 0; layer < g.layers; ++layer) {
        for (auto role : shared) { add(role, layer); }
        if (is_full_attention_layer(g, layer)) {
            for (auto role : attention) { add(role, layer); }
        } else {
            for (auto role : gdn) { add(role, layer); }
        }
    }
    return out;
}

// Complete frozen-config metadata for the FULL trunk fixture (values mirror
// Qwen35Geometry / the reference checkpoint). `nextn` selects the combined
// trunk+draft declaration (block_count=33, nextn_predict_layers=1) vs. the
// default trunk-only declaration (block_count=32, no nextn key).
[[nodiscard]] luisa::vector<std::pair<luisa::string, luisa::vector<std::byte>>>
full_qwen35_metas(bool nextn) {
    const Qwen35Geometry g{};
    luisa::vector<std::pair<luisa::string, luisa::vector<std::byte>>> m;
    const auto u = [&](const char *key, uint64_t v) {
        m.emplace_back(key, meta_u32_blob(static_cast<uint32_t>(v)));
    };
    m.emplace_back("general.architecture", meta_string_blob("qwen35"));
    u("qwen35.block_count", g.layers + (nextn ? 1u : 0u));
    u("qwen35.context_length", g.context_length);
    u("qwen35.embedding_length", g.hidden);
    u("qwen35.feed_forward_length", g.ffn);
    u("qwen35.attention.head_count", g.q_heads);
    u("qwen35.attention.head_count_kv", g.kv_heads);
    u("qwen35.attention.key_length", g.head_dim);
    u("qwen35.attention.value_length", g.head_dim);
    u("qwen35.rope.dimension_count", g.rope_dims);
    u("qwen35.ssm.conv_kernel", g.conv_kernel);
    u("qwen35.ssm.state_size", g.gdn_head_dim);
    u("qwen35.ssm.group_count", g.gdn_qk_heads);
    u("qwen35.ssm.time_step_rank", g.gdn_v_heads);
    u("qwen35.ssm.inner_size", g.gdn_z_n);
    u("qwen35.full_attention_interval", g.full_attention_interval);
    m.emplace_back("qwen35.rope.freq_base", meta_f32_blob(static_cast<float>(g.rope_theta)));
    m.emplace_back("qwen35.attention.layer_norm_rms_epsilon",
                   meta_f32_blob(static_cast<float>(g.rms_eps)));
    const uint32_t sections[] = {11u, 11u, 10u, 0u};
    m.emplace_back("qwen35.rope.dimension_sections", meta_u32_array_blob(sections));
    m.emplace_back("tokenizer.ggml.model", meta_string_blob("gpt2"));
    m.emplace_back("tokenizer.ggml.pre", meta_string_blob("qwen35"));
    u("tokenizer.ggml.eos_token_id", 248044u);
    u("tokenizer.ggml.padding_token_id", 248044u);
    if (nextn) { u(kMtpNextnMetadataKey.data(), 1u); }
    return m;
}

[[nodiscard]] luisa::vector<std::pair<luisa::string, luisa::vector<std::byte>>>
qwen35_metas(uint64_t block_count, bool with_nextn, uint64_t nextn) {
    luisa::vector<std::pair<luisa::string, luisa::vector<std::byte>>> m;
    m.emplace_back("general.architecture", meta_string_blob("qwen35"));
    m.emplace_back("qwen35.block_count", meta_u32_blob(static_cast<uint32_t>(block_count)));
    if (with_nextn) {
        m.emplace_back(luisa::string{kMtpNextnMetadataKey},
                       meta_u32_blob(static_cast<uint32_t>(nextn)));
    }
    return m;
}

}// namespace

[[nodiscard]] Status mtp_self_test(luisa::vector<luisa::string> *report) {
    MtpSelfTestCtx ctx{report, 0, 0};
    const auto fail = [&](luisa::string msg) -> Status {
        ctx.note("FAIL: " + msg);
        return Status::error(StatusCode::internal_error, "mtp self-test: " + msg);
    };

    std::error_code ec;
    const auto dir = luisa::filesystem::temp_directory_path(ec) / "lcqwen_mtp_selftest";
    if (ec) { return fail("cannot resolve temp directory"); }
    luisa::filesystem::create_directories(dir, ec);
    if (ec) { return fail("cannot create temp directory"); }
    struct Cleanup {
        luisa::filesystem::path p;
        ~Cleanup() {
            std::error_code e;
            luisa::filesystem::remove_all(p, e);
        }
    } cleanup{dir};

    // --- case 1: trunk-only slice, no nextn metadata, no blk.32 (the current
    //     real checkpoint situation in miniature) -----------------------------
    {
        luisa::vector<MtpFixtureTensor> tensors = {
            {"token_embd.weight", 30u, {2560, 248320}},
            {"blk.0.attn_norm.weight", 0u, {2560}},
            {"blk.31.attn_q.weight", 30u, {2560, 8192}},
            {"output_norm.weight", 0u, {2560}},
        };
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "no_mtp", qwen35_metas(32, false, 0), tensors, r)) {
            return fail("no_mtp fixture");
        }
        ctx.expect(!r.nextn_metadata_present, "no_mtp: nextn metadata absent");
        ctx.expect(r.detected_tensor_count == 0, "no_mtp: zero blk.32 tensors");
        ctx.expect(r.required_missing_count == kMtpRequiredTensorCount,
                   "no_mtp: all 15 required tensors missing");
        ctx.expect(!r.available && !r.draft_complete && !r.benchmark_feasible,
                   "no_mtp: unavailable");
        ctx.expect(!r.benchmarked && !r.benchmark_passed && !r.accelerates && !r.default_enabled,
                   "no_mtp: policy flags all false");
        ctx.expect(r.reason.find("missing") != luisa::string::npos,
                   "no_mtp: reason mentions missing weights");
        ctx.note("fixture no_mtp ok (missing-weights verdict)");
    }

    // --- case 2: complete canonical draft + nextn=1, block_count=33 ----------
    {
        const auto tensors = canonical_draft_tensors();
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "complete", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("complete fixture");
        }
        ctx.expect(r.nextn_metadata_present && r.nextn_metadata_valid && r.nextn_predict_layers == 1,
                   "complete: nextn metadata parsed");
        ctx.expect(r.detected_tensor_count == kMtpRequiredTensorCount,
                   "complete: 15 canonical tensors detected");
        ctx.expect(r.required_ok_count == kMtpRequiredTensorCount &&
                       r.required_missing_count == 0 && r.required_malformed_count == 0,
                   "complete: all required tensors ok");
        ctx.expect(r.draft_complete && r.available, "complete: draft complete/available");
        // The fixture has no trunk, so the A/B benchmark must NOT be claimed.
        ctx.expect(!r.target_trunk_ok && !r.benchmark_feasible,
                   "complete: no trunk => benchmark_feasible false");
        ctx.expect(!r.benchmarked && !r.benchmark_passed && !r.accelerates && !r.default_enabled,
                   "complete: weights alone never enable MTP");
        ctx.expect(r.reason.find("none has been measured") != luisa::string::npos ||
                       r.reason.find("does not validate") != luisa::string::npos,
                   "complete: reason stays honest about benchmarking");
        ctx.note("fixture complete ok (available, but unbenchmarked => disabled)");
    }

    // --- case 3: one required tensor missing ---------------------------------
    {
        auto tensors = canonical_draft_tensors();
        tensors.erase(tensors.begin());// drop eh_proj
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "partial", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("partial fixture");
        }
        ctx.expect(r.required_missing_count == 1 && r.required_ok_count == 14,
                   "partial: exactly one missing");
        ctx.expect(!r.draft_complete && !r.available && !r.benchmark_feasible,
                   "partial: fail closed");
        ctx.expect(r.reason.find("eh_proj") != luisa::string::npos,
                   "partial: reason names the missing tensor");
        ctx.note("fixture partial ok (incomplete inventory fails closed)");
    }

    // --- case 4: wrong shape --------------------------------------------------
    {
        auto tensors = canonical_draft_tensors();
        for (auto &t : tensors) {
            if (t.name == "blk.32.eh_proj.weight") { t.dims = {2560, 2560}; }
        }
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "badshape", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("badshape fixture");
        }
        ctx.expect(r.required_malformed_count == 1 && !r.available,
                   "badshape: malformed shape fails closed");
        ctx.expect(r.reason.find("shape") != luisa::string::npos,
                   "badshape: reason mentions shape");
        ctx.note("fixture badshape ok");
    }

    // --- case 5a: wrong dtype the reader CAN open (norm stored BF16) ---------
    {
        auto tensors = canonical_draft_tensors();
        for (auto &t : tensors) {
            if (t.name == "blk.32.enorm.weight") { t.type = 30u; }// BF16 norm: rejected
        }
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "baddtype", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("baddtype fixture");
        }
        ctx.expect(r.required_malformed_count == 1 && !r.available,
                   "baddtype: BF16 norm fails closed");
        ctx.expect(r.reason.find("F32") != luisa::string::npos,
                   "baddtype: reason names the F32 requirement");
        ctx.note("fixture baddtype ok (BF16 norm detected as malformed)");
    }

    // --- case 5b: dtype the reader itself refuses (F16 matrix) ----------------
    {
        auto tensors = canonical_draft_tensors();
        for (auto &t : tensors) {
            if (t.name == "blk.32.ffn_gate.weight") { t.type = 1u; }// F16: unsupported
        }
        MtpEligibilityReport r;
        bool open_rejected = false;
        if (!probe_fixture(ctx, dir, "f16dtype", qwen35_metas(33, true, 1), tensors, r,
                           &open_rejected)) {
            // The reader rejects F16 at open time: fail closed even earlier
            // than the probe. That is the correct outcome.
            ctx.expect(open_rejected, "f16dtype: rejected at GGUF open");
            ctx.note("fixture f16dtype rejected at GGUF open (fail closed) -- ok");
        } else {
            ctx.expect(r.required_malformed_count >= 1 && !r.available,
                       "f16dtype: dtype mismatch fails closed");
            ctx.note("fixture f16dtype ok (dtype mismatch detected)");
        }
    }

    // --- case 5c: Q8_0 draft matrix (reader opens it; probe must reject) ------
    {
        auto tensors = canonical_draft_tensors();
        for (auto &t : tensors) {
            if (t.name == "blk.32.ffn_gate.weight") { t.type = 8u; }// Q8_0
        }
        // Keep the canonical shape (element count is a multiple of 32, as
        // Q8_0 requires); the fixture payload is zero-filled and large
        // enough for the 34-bytes-per-32-elements block layout. The probe
        // must reject Q8_0 on dtype grounds before shape even matters.
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "q8dtype", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("q8dtype fixture");
        }
        ctx.expect(r.required_malformed_count == 1 && !r.available,
                   "q8dtype: Q8_0 draft matrix fails closed");
        ctx.expect(r.reason.find("Q8_0") != luisa::string::npos,
                   "q8dtype: reason names Q8_0");
        ctx.note("fixture q8dtype ok (Q8_0 is not silently accepted)");
    }

    // --- case 6: unexpected blk.32.* tensor (GDN leftover) --------------------
    {
        auto tensors = canonical_draft_tensors();
        tensors.emplace_back(MtpFixtureTensor{"blk.32.ssm_conv1d.weight", 0u, {4, 8192}});
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "unexpected", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("unexpected fixture");
        }
        ctx.expect(r.unexpected_tensor_names.size() == 1 && !r.available,
                   "unexpected: extra blk.32 tensor fails closed");
        ctx.expect(r.reason.find("unexpected") != luisa::string::npos,
                   "unexpected: reason mentions unexpected tensors");
        ctx.note("fixture unexpected ok");
    }

    // --- case 7: tensors present but nextn metadata absent --------------------
    {
        const auto tensors = canonical_draft_tensors();
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "nometa", qwen35_metas(32, false, 0), tensors, r)) {
            return fail("nometa fixture");
        }
        ctx.expect(r.detected_tensor_count == kMtpRequiredTensorCount && !r.nextn_metadata_present,
                   "nometa: tensors detected without metadata");
        ctx.expect(!r.available && r.reason.find("metadata key is") != luisa::string::npos,
                   "nometa: undeclared NextN fails closed");
        ctx.note("fixture nometa ok");
    }

    // --- case 8: nextn metadata != 1 ------------------------------------------
    {
        const auto tensors = canonical_draft_tensors();
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "nextn2", qwen35_metas(34, true, 2), tensors, r)) {
            return fail("nextn2 fixture");
        }
        ctx.expect(r.nextn_predict_layers == 2 && !r.available,
                   "nextn2: two-layer declaration fails closed");
        ctx.note("fixture nextn2 ok");
    }

    // --- case 9: block_count mismatch ------------------------------------------
    {
        const auto tensors = canonical_draft_tensors();
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "badblocks", qwen35_metas(32, true, 1), tensors, r)) {
            return fail("badblocks fixture");
        }
        ctx.expect(!r.available && r.reason.find("block_count") != luisa::string::npos,
                   "badblocks: block_count=32 with nextn=1 fails closed");
        ctx.note("fixture badblocks ok");
    }

    // --- case 10: non-qwen35 architecture ---------------------------------------
    {
        luisa::vector<std::pair<luisa::string, luisa::vector<std::byte>>> metas = {
            {"general.architecture", meta_string_blob("llama")},
        };
        luisa::vector<MtpFixtureTensor> tensors = {{"blk.32.enorm.weight", 0u, {2560}}};
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "otherarch", metas, tensors, r)) {
            return fail("otherarch fixture");
        }
        ctx.expect(!r.architecture_is_qwen35 && !r.available,
                   "otherarch: foreign architecture fails closed");
        ctx.note("fixture otherarch ok");
    }

    // --- case 11: malformed OPTIONAL tensors (present but broken) ------------
    // Optional means "may be absent", never "may be broken": a wrong-shape
    // blk.32.token_embd / wrong-dtype blk.32.output must fail draft_complete.
    {
        auto tensors = canonical_draft_tensors();
        tensors.emplace_back(MtpFixtureTensor{"blk.32.token_embd.weight", 30u, {2560, 1000}});
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "badoptional_shape", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("badoptional_shape fixture");
        }
        ctx.expect(r.optional_present_count == 1 && r.optional_malformed_count == 1,
                   "badoptional_shape: optional tensor present but malformed");
        ctx.expect(!r.draft_complete && !r.available && !r.benchmark_feasible,
                   "badoptional_shape: malformed optional token_embd fails closed");
        ctx.expect(r.reason.find("optional") != luisa::string::npos &&
                       r.reason.find("blk.32.token_embd.weight") != luisa::string::npos,
                   "badoptional_shape: reason names the malformed optional tensor");
        ctx.note("fixture badoptional_shape ok (broken optional fails closed)");
    }
    {
        auto tensors = canonical_draft_tensors();
        // Q8_0 dedicated LM head: reader opens it, probe must reject it.
        tensors.emplace_back(MtpFixtureTensor{"blk.32.output.weight", 8u, {2560, 248320}});
        MtpEligibilityReport r;
        if (!probe_fixture(ctx, dir, "badoptional_dtype", qwen35_metas(33, true, 1), tensors, r)) {
            return fail("badoptional_dtype fixture");
        }
        ctx.expect(r.optional_present_count == 1 && r.optional_malformed_count == 1,
                   "badoptional_dtype: optional Q8_0 head detected as malformed");
        ctx.expect(!r.draft_complete && !r.available && !r.benchmark_feasible,
                   "badoptional_dtype: malformed optional output fails closed");
        ctx.expect(r.reason.find("blk.32.output.weight") != luisa::string::npos,
                   "badoptional_dtype: reason names the malformed optional tensor");
        ctx.note("fixture badoptional_dtype ok (Q8_0 optional head fails closed)");
    }

    // --- case 12: FULL combined file -- complete valid 426-tensor trunk PLUS
    //     the 15 draft tensors. The positive path must be real: default load
    //     rejects the combined file, opt-in load accepts it, and the probe
    //     reports target_trunk_ok=true / benchmark_feasible=true (weights
    //     feasibility ONLY; every measured/default flag stays false). -------
    {
        auto tensors = full_trunk_tensors();
        ctx.expect(tensors.size() == 426, "combined: full trunk fixture has exactly 426 tensors");
        const auto draft = canonical_draft_tensors();
        tensors.insert(tensors.end(), draft.begin(), draft.end());
        ctx.note("combined_full: streaming ~8.7 GB zero-payload fixture (real frozen shapes)...");
        luisa::filesystem::path path;
        if (!write_streamed_fixture(ctx, dir, "combined_full", full_qwen35_metas(true), tensors, path)) {
            return fail("combined_full fixture write");
        }
        auto reader = GgufReader::open(path);
        if (!reader.ok()) { return fail("combined_full open: " + reader.status.message); }
        const auto &rd = **reader.value;
        ctx.expect(rd.tensors().size() == 441, "combined: reader sees 426 + 15 tensors");
        // The DEFAULT engine contract must reject the combined file...
        ctx.expect(!Qwen35Model::load(rd).ok(),
                   "combined: default Qwen35Model::load rejects trunk+draft (exact 426 contract)");
        // ...while the opt-in mode validates the full trunk bijection.
        auto trunk = Qwen35Model::load_with_nextn(rd);
        ctx.expect(trunk.ok(), "combined: opt-in load_with_nextn validates the combined trunk");
        if (trunk.ok()) {
            ctx.expect(trunk.value->tensors().size() == 426,
                       "combined: opt-in trunk resolves exactly the 426 frozen tensors");
        }
        const auto r = probe_mtp_eligibility(rd);
        ctx.expect(r.total_tensor_count == 441 && r.detected_tensor_count == 15,
                   "combined: probe sees the full combined inventory");
        ctx.expect(r.required_ok_count == kMtpRequiredTensorCount &&
                       r.required_missing_count == 0 && r.required_malformed_count == 0 &&
                       r.optional_malformed_count == 0 && r.unexpected_tensor_names.empty(),
                   "combined: draft inventory fully ok");
        ctx.expect(r.draft_complete && r.available, "combined: draft_complete=true");
        ctx.expect(r.target_trunk_ok, "combined: target_trunk_ok=true (FULL valid trunk)");
        ctx.expect(r.benchmark_feasible, "combined: benchmark_feasible=true (weights feasibility)");
        ctx.expect(!r.benchmarked && !r.benchmark_passed && !r.accelerates && !r.default_enabled,
                   "combined: measured/default flags all stay false");
        ctx.expect(r.reason.find("none has been measured") != luisa::string::npos,
                   "combined: reason stays honest (feasibility, not acceleration)");
        ctx.note("fixture combined_full ok (426 trunk + 15 draft => benchmark_feasible, still disabled)");
    }

    // --- case 12b: the same FULL trunk alone stays the default contract ------
    {
        const auto tensors = full_trunk_tensors();
        luisa::filesystem::path path;
        if (!write_streamed_fixture(ctx, dir, "trunk_full", full_qwen35_metas(false), tensors, path)) {
            return fail("trunk_full fixture write");
        }
        auto reader = GgufReader::open(path);
        if (!reader.ok()) { return fail("trunk_full open: " + reader.status.message); }
        const auto &rd = **reader.value;
        ctx.expect(Qwen35Model::load(rd).ok(),
                   "trunk_full: default load accepts the exact 426/block_count=32 trunk");
        ctx.expect(!Qwen35Model::load_with_nextn(rd).ok(),
                   "trunk_full: opt-in mode rejects a trunk without the declared NextN extension");
        const auto r = probe_mtp_eligibility(rd);
        ctx.expect(!r.available && !r.draft_complete && !r.benchmark_feasible,
                   "trunk_full: probe reports missing weights");
        ctx.expect(r.reason.find("missing") != luisa::string::npos,
                   "trunk_full: reason is missing weights");
        ctx.note("fixture trunk_full ok (default 426 contract unchanged)");
    }

    // --- case 12c: combined file with a NON-vetted extra tensor --------------
    // The probe's unexpected-tensor check fires first; additionally the
    // opt-in trunk loader itself must refuse extras outside the vetted set.
    {
        auto tensors = full_trunk_tensors();
        const auto draft = canonical_draft_tensors();
        tensors.insert(tensors.end(), draft.begin(), draft.end());
        tensors.emplace_back(MtpFixtureTensor{"blk.32.ssm_conv1d.weight", 0u, {4, 8192}});
        luisa::filesystem::path path;
        if (!write_streamed_fixture(ctx, dir, "combined_badextra", full_qwen35_metas(true), tensors, path)) {
            return fail("combined_badextra fixture write");
        }
        auto reader = GgufReader::open(path);
        if (!reader.ok()) { return fail("combined_badextra open: " + reader.status.message); }
        const auto r = probe_mtp_eligibility(**reader.value);
        ctx.expect(r.unexpected_tensor_names.size() == 1 &&
                       !r.draft_complete && !r.available && !r.benchmark_feasible,
                   "combined_badextra: non-vetted blk.32 extra fails closed");
        ctx.note("fixture combined_badextra ok");
    }

    // --- real checkpoint (when present): the honest immediate verdict ---------
    {
        luisa::filesystem::path model_path;
        const bool have_path = luisa::path_from_utf8("data/Qwen3.5-4B-BF16.gguf", model_path);
        std::error_code exists_ec;
        const bool exists = have_path && luisa::filesystem::exists(model_path, exists_ec) && !exists_ec;
        if (!exists) {
            ctx.note("real checkpoint data/Qwen3.5-4B-BF16.gguf not present -- skipped");
        } else {
            auto reader = GgufReader::open(model_path);
            if (!reader.ok()) { return fail("real GGUF open: " + reader.status.message); }
            const auto r = probe_mtp_eligibility(**reader.value);
            for (auto &line : r.report_lines()) { ctx.note(line); }
            ctx.expect(r.total_tensor_count == 426, "real: 426 tensors");
            ctx.expect(!r.nextn_metadata_present, "real: no qwen35.nextn_predict_layers metadata");
            ctx.expect(r.detected_tensor_count == 0, "real: no blk.32.* tensors");
            ctx.expect(!r.available && !r.draft_complete && !r.benchmark_feasible,
                       "real: available=false");
            ctx.expect(!r.benchmarked && !r.benchmark_passed && !r.accelerates && !r.default_enabled,
                       "real: benchmarked=false accelerates=false default_enabled=false");
            ctx.expect(r.reason.find("missing") != luisa::string::npos,
                       "real: reason is missing weights");
            // The trunk itself must still validate (the probe must not have
            // broken or misread the reference inventory).
            ctx.expect(Qwen35Model::load(**reader.value).ok(),
                       "real: frozen 426-tensor trunk still validates");
            ctx.note("real checkpoint probe ok (honest missing-weights verdict)");
        }
    }

    ctx.note(luisa::format("mtp_self_test: {} check(s), {} failure(s)", ctx.checks, ctx.failures));
    if (ctx.failures != 0) {
        return Status::error(StatusCode::internal_error,
                             luisa::format("mtp self-test: {} check(s) failed", ctx.failures));
    }
    return Status::success();
}

}// namespace luisa::qwen
