#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#if defined(__wasm_simd128__) || defined(__wasm__)
#include <wasm_simd128.h>
#endif

#ifndef EMSCRIPTEN_KEEPALIVE
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif
#endif

// Hyperparameters for ultra-lightweight Narrative-Nano inference
constexpr size_t VOCAB_SIZE = 256;
constexpr size_t HIDDEN_DIM = 64;   // Multiple of 4 for 128-bit SIMD (4 x float32)
constexpr size_t NUM_LAYERS = 2;

struct NanoModel {
    bool initialized = false;
    std::vector<float> token_embedding_table; // [VOCAB_SIZE * HIDDEN_DIM]
    std::vector<float> w_linear;              // [HIDDEN_DIM * HIDDEN_DIM]
    std::vector<float> w_head;                // [HIDDEN_DIM * VOCAB_SIZE]
    std::vector<float> hidden_state;          // [HIDDEN_DIM]
};

static NanoModel g_model;

// WebAssembly SIMD128 Dot-Product: computes dot(a, b) where length is multiple of 4
static inline float simd_dot_product(const float* a, const float* b, size_t len) {
#if defined(__wasm_simd128__)
    v128_t acc = wasm_f32x4_splat(0.0f);
    for (size_t i = 0; i < len; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(va, vb));
    }
    float s0 = wasm_f32x4_extract_lane(acc, 0);
    float s1 = wasm_f32x4_extract_lane(acc, 1);
    float s2 = wasm_f32x4_extract_lane(acc, 2);
    float s3 = wasm_f32x4_extract_lane(acc, 3);
    return s0 + s1 + s2 + s3;
#else
    float acc = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

// SIMD-accelerated Matrix-Vector Multiplication: out = W * in
// W is of shape [out_dim, in_dim] row-major. in is of shape [in_dim]
static inline void simd_gemv(float* out, const float* W, const float* in, size_t out_dim, size_t in_dim) {
    for (size_t r = 0; r < out_dim; ++r) {
        out[r] = simd_dot_product(&W[r * in_dim], in, in_dim);
    }
}

// In-place ReLU / activation
static inline void simd_relu(float* vec, size_t len) {
#if defined(__wasm_simd128__)
    v128_t zero = wasm_f32x4_splat(0.0f);
    for (size_t i = 0; i < len; i += 4) {
        v128_t v = wasm_v128_load(&vec[i]);
        v128_t max_v = wasm_f32x4_pmax(v, zero);
        wasm_v128_store(&vec[i], max_v);
    }
#else
    for (size_t i = 0; i < len; ++i) {
        if (vec[i] < 0.0f) vec[i] = 0.0f;
    }
#endif
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int _init_model(const uint8_t* weight_data, size_t size) {
    if (!weight_data || size == 0) {
        return -1;
    }

    g_model.token_embedding_table.resize(VOCAB_SIZE * HIDDEN_DIM);
    g_model.w_linear.resize(HIDDEN_DIM * HIDDEN_DIM);
    g_model.w_head.resize(HIDDEN_DIM * VOCAB_SIZE);
    g_model.hidden_state.assign(HIDDEN_DIM, 0.0f);

    size_t expected_elements = g_model.token_embedding_table.size() +
                              g_model.w_linear.size() +
                              g_model.w_head.size();
    size_t expected_bytes = expected_elements * sizeof(float);

    if (size >= expected_bytes) {
        const float* fptr = reinterpret_cast<const float*>(weight_data);
        size_t offset = 0;
        std::memcpy(g_model.token_embedding_table.data(), fptr + offset, g_model.token_embedding_table.size() * sizeof(float));
        offset += g_model.token_embedding_table.size();
        std::memcpy(g_model.w_linear.data(), fptr + offset, g_model.w_linear.size() * sizeof(float));
        offset += g_model.w_linear.size();
        std::memcpy(g_model.w_head.data(), fptr + offset, g_model.w_head.size() * sizeof(float));
    } else {
        // Fallback initialization: populate deterministic pseudo-weights from byte stream
        for (size_t i = 0; i < g_model.token_embedding_table.size(); ++i) {
            g_model.token_embedding_table[i] = static_cast<float>(weight_data[i % size] - 128) / 128.0f;
        }
        for (size_t i = 0; i < g_model.w_linear.size(); ++i) {
            g_model.w_linear[i] = static_cast<float>(weight_data[(i * 3 + 17) % size] - 128) / 128.0f;
        }
        for (size_t i = 0; i < g_model.w_head.size(); ++i) {
            g_model.w_head[i] = static_cast<float>(weight_data[(i * 7 + 31) % size] - 128) / 128.0f;
        }
    }

    g_model.initialized = true;
    return 0; // Success
}

EMSCRIPTEN_KEEPALIVE
int _forward_step(int token_id, float* output_logits) {
    if (!g_model.initialized || !output_logits) {
        return -1;
    }

    token_id = std::max(0, std::min<int>(token_id, static_cast<int>(VOCAB_SIZE - 1)));

    // 1. Embedding lookup
    const float* emb = &g_model.token_embedding_table[token_id * HIDDEN_DIM];
    std::vector<float> layer_in(HIDDEN_DIM);
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        layer_in[i] = g_model.hidden_state[i] + emb[i];
    }

    // 2. Transformer / MLP Feed-Forward with Wasm SIMD128
    std::vector<float> layer_out(HIDDEN_DIM, 0.0f);
    simd_gemv(layer_out.data(), g_model.w_linear.data(), layer_in.data(), HIDDEN_DIM, HIDDEN_DIM);
    simd_relu(layer_out.data(), HIDDEN_DIM);

    // Update recurrent hidden state
    for (size_t i = 0; i < HIDDEN_DIM; ++i) {
        g_model.hidden_state[i] = 0.5f * g_model.hidden_state[i] + 0.5f * layer_out[i];
    }

    // 3. Final Projection to Logits using SIMD GEMV: [VOCAB_SIZE, HIDDEN_DIM] x [HIDDEN_DIM]
    simd_gemv(output_logits, g_model.w_head.data(), g_model.hidden_state.data(), VOCAB_SIZE, HIDDEN_DIM);

    return 0; // Success
}

EMSCRIPTEN_KEEPALIVE
void _free_model() {
    g_model.token_embedding_table.clear();
    g_model.token_embedding_table.shrink_to_fit();
    g_model.w_linear.clear();
    g_model.w_linear.shrink_to_fit();
    g_model.w_head.clear();
    g_model.w_head.shrink_to_fit();
    g_model.hidden_state.clear();
    g_model.hidden_state.shrink_to_fit();
    g_model.initialized = false;
}

} // extern "C"
