#ifdef ML_USE_CUDA

#include "ml_lib/cuda/gpu-transformer.h"
#include "ml_lib/cuda/cuda-matrix.h"
#include "ml_lib/cuda/cuda-kernels.h"
#include "ml_lib/models/transformer.h"
#include "ml_lib/core/transformer-block.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

// This was generated using claude


// ============================================================
// Local cuBLAS helpers (templated, row-major)
// float → cublasSgemm, __half → cublasGemmEx (Tensor Core fp16)
// ============================================================

namespace {

static cublasHandle_t& cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) CUBLAS_CHECK(cublasCreate(&h));
    return h;
}

template<typename T>
T make_val(float x);
template<> float make_val<float>(float x) { return x; }
template<> __half make_val<__half>(float x) { return __float2half(x); }

// Dispatch gemm: cublasSgemm for float, cublasGemmEx for __half
template<typename T>
void gemm_dispatch(cublasOperation_t opA, cublasOperation_t opB,
                   int M_cublas, int N_cublas, int K_cublas,
                   const T* A, int lda,
                   const T* B, int ldb,
                   T* C, int ldc,
                   float alpha_val = 1.0f) {
    float alpha = alpha_val, beta = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle(),
            opA, opB,
            M_cublas, N_cublas, K_cublas, &alpha,
            A, lda, B, ldb, &beta, C, ldc));
    } else {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
            opA, opB,
            M_cublas, N_cublas, K_cublas,
            &alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            &beta,
            C, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

// C(M,N) = alpha * A(M,K) * B(K,N), all contiguous row-major
template<typename T>
void gemm(const T* A, const T* B, T* C, int M, int N, int K, float alpha = 1.0f) {
    gemm_dispatch<T>(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, B, N, A, K, C, N, alpha);
}

// C(M,N) = A(M,K) * B^T  where B is stored as (N,K) row-major
template<typename T>
void gemm_ABt(const T* A, int stride_a,
              const T* B, int stride_b,
              T* C, int stride_c,
              int M, int N, int K) {
    gemm_dispatch<T>(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     B, stride_b, A, stride_a, C, stride_c);
}

// C(M,N) = A(M,K) * B(K,N) with explicit strides
template<typename T>
void gemm_strided(const T* A, int stride_a,
                  const T* B, int stride_b,
                  T* C, int stride_c,
                  int M, int N, int K) {
    gemm_dispatch<T>(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     B, stride_b, A, stride_a, C, stride_c);
}

} // anonymous namespace

// ============================================================
// GpuScratch — pre-allocate all activation buffers
// ============================================================

template<typename T>
void GpuScratch<T>::allocate(int max_seq_len, int embed_dim, int ff_dim,
                              int kv_dim, int vocab_size) {
    x        = CudaMatrix<T>(max_seq_len, embed_dim);
    norm_out = CudaMatrix<T>(max_seq_len, embed_dim);
    Q        = CudaMatrix<T>(max_seq_len, embed_dim);
    K        = CudaMatrix<T>(max_seq_len, kv_dim);
    V        = CudaMatrix<T>(max_seq_len, kv_dim);
    attn_out = CudaMatrix<T>(max_seq_len, embed_dim);
    scores   = CudaMatrix<T>(max_seq_len, max_seq_len);
    gate_out = CudaMatrix<T>(max_seq_len, ff_dim);
    up_out   = CudaMatrix<T>(max_seq_len, ff_dim);
    silu_out = CudaMatrix<T>(max_seq_len, ff_dim);
    ffn_out  = CudaMatrix<T>(max_seq_len, embed_dim);
    logits   = CudaMatrix<T>(1, vocab_size);
    last_row = CudaMatrix<T>(1, embed_dim);
}

// ============================================================
// Transfer CPU model weights to GPU (with type conversion)
// ============================================================

template<typename GpuT, typename CpuT>
GpuTransformerWeights<GpuT> transfer_weights_to_gpu(Transformer<CpuT>& model) {
    GpuTransformerWeights<GpuT> w;

    // Embedding table
    w.token_embedding = CudaMatrix<GpuT>::from_cpu_float(model.getEmbedding().getWeights());

    // Per-layer weights
    auto& blocks = model.getBlocks();
    w.layers.reserve(blocks.size());

    for (auto& block : blocks) {
        GpuLayerWeights<GpuT> lw;
        auto& attn = block->getAttention();

        // Attention projections
        lw.W_q = CudaMatrix<GpuT>::from_cpu_float(attn.getWq());
        lw.W_k = CudaMatrix<GpuT>::from_cpu_float(attn.getWk());
        lw.W_v = CudaMatrix<GpuT>::from_cpu_float(attn.getWv());
        lw.W_o = CudaMatrix<GpuT>::from_cpu_float(attn.getWo());

        // RoPE tables
        if (attn.isRoPEEnabled()) {
            lw.rope_cos = CudaMatrix<GpuT>::from_cpu_float(attn.getRopeCos());
            lw.rope_sin = CudaMatrix<GpuT>::from_cpu_float(attn.getRopeSin());
        }

        // RMS norm gammas
        lw.rms1_gamma = CudaMatrix<GpuT>::from_cpu_float(block->getRMSNorm1()->getGamma());
        lw.rms2_gamma = CudaMatrix<GpuT>::from_cpu_float(block->getRMSNorm2()->getGamma());

        // FFN weights (gated SiLU: gate, up, down)
        lw.ff_gate_w = CudaMatrix<GpuT>::from_cpu_float(block->getFFGate()->getWeights());
        lw.ff_up_w   = CudaMatrix<GpuT>::from_cpu_float(block->getFF1().getWeights());
        lw.ff_down_w = CudaMatrix<GpuT>::from_cpu_float(block->getFF2().getWeights());

        w.layers.push_back(std::move(lw));
    }

    // Output RMS norm
    w.output_rms_gamma = CudaMatrix<GpuT>::from_cpu_float(model.getOutputRMSNorm()->getGamma());

    // Output projection
    w.output_proj_w = CudaMatrix<GpuT>::from_cpu_float(model.getOutputProjection().getWeights());

    return w;
}

// ============================================================
// GPU forward — batched prefill (all prompt tokens at once)
// ============================================================

template<typename T>
void gpu_forward_prefill(const std::vector<int>& tokens,
                         const GpuTransformerWeights<T>& weights,
                         GpuKVCache<T>& kv_cache,
                         GpuScratch<T>& scratch,
                         int num_heads, int num_kv_heads, int head_dim,
                         int embed_dim, int ff_dim, float rms_eps) {
    int seq_len = static_cast<int>(tokens.size());
    int kv_dim = num_kv_heads * head_dim;
    int heads_per_group = num_heads / num_kv_heads;

    // Upload token IDs → embedding lookup
    int* d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(),
                           seq_len * sizeof(int), cudaMemcpyHostToDevice));
    cuda_kernels::embedding_lookup(d_tokens, weights.token_embedding.data(),
                                    scratch.x.data(), seq_len, embed_dim);
    CUDA_CHECK(cudaFree(d_tokens));

    int num_layers = static_cast<int>(weights.layers.size());
    for (int l = 0; l < num_layers; l++) {
        const auto& lw = weights.layers[l];

        // --- Attention sub-block ---

        // 1. Pre-norm
        cuda_kernels::rms_norm(scratch.x.data(), lw.rms1_gamma.data(),
                                scratch.norm_out.data(), seq_len, embed_dim, rms_eps);

        // 2. Q/K/V projections (Q is pre-scaled by 1/sqrt(head_dim) to
        //    keep attention scores in fp16 range)
        float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        gemm(scratch.norm_out.data(), lw.W_q.data(),
             scratch.Q.data(), seq_len, embed_dim, embed_dim, attn_scale);
        gemm(scratch.norm_out.data(), lw.W_k.data(),
             scratch.K.data(), seq_len, kv_dim, embed_dim);
        gemm(scratch.norm_out.data(), lw.W_v.data(),
             scratch.V.data(), seq_len, kv_dim, embed_dim);

        // 3. RoPE (start_pos=0 for prefill)
        if (!lw.rope_cos.empty()) {
            cuda_kernels::rope(scratch.Q.data(), scratch.K.data(),
                                lw.rope_cos.data(), lw.rope_sin.data(),
                                seq_len, num_heads, num_kv_heads, head_dim, 0);
        }

        // 4. Write K, V into KV cache at positions 0..seq_len-1
        kv_cache.layers[l].K.write_rows(0, scratch.K.data(), seq_len, kv_dim);
        kv_cache.layers[l].V.write_rows(0, scratch.V.data(), seq_len, kv_dim);

        // 5. Per-head attention (Q already pre-scaled, so scale=1.0 here)
        T one = make_val<T>(1.0f);

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_group;

            const T* Q_h = scratch.Q.data() + h * head_dim;
            const T* K_h = kv_cache.layers[l].K.data() + kv_h * head_dim;
            const T* V_h = kv_cache.layers[l].V.data() + kv_h * head_dim;
            T* attn_h    = scratch.attn_out.data() + h * head_dim;

            // scores(seq_len, seq_len) = Q_h * K_h^T (already scaled)
            gemm_ABt(Q_h, embed_dim, K_h, kv_dim,
                     scratch.scores.data(), seq_len,
                     seq_len, seq_len, head_dim);

            // Causal mask only (offset=0: position i sees j <= i)
            cuda_kernels::scale_causal_mask(
                scratch.scores.data(), seq_len, seq_len, one, 0);

            // Softmax (in-place)
            cuda_kernels::softmax(scratch.scores.data(), scratch.scores.data(),
                                   seq_len, seq_len);

            // attn_h(seq_len, head_dim) = softmax * V_h
            gemm_strided(scratch.scores.data(), seq_len,
                         V_h, kv_dim,
                         attn_h, embed_dim,
                         seq_len, head_dim, seq_len);
        }

        // 6. Output projection: ffn_out = attn_out * W_o
        gemm(scratch.attn_out.data(), lw.W_o.data(),
             scratch.ffn_out.data(), seq_len, embed_dim, embed_dim);

        // 7. Residual: x += attn_output
        cuda_kernels::add_inplace(scratch.x.data(), scratch.ffn_out.data(),
                                   seq_len * embed_dim);

        // --- FFN sub-block ---

        // 8. Second pre-norm
        cuda_kernels::rms_norm(scratch.x.data(), lw.rms2_gamma.data(),
                                scratch.norm_out.data(), seq_len, embed_dim, rms_eps);

        // 9. Gated FFN projections
        gemm(scratch.norm_out.data(), lw.ff_gate_w.data(),
             scratch.gate_out.data(), seq_len, ff_dim, embed_dim);
        gemm(scratch.norm_out.data(), lw.ff_up_w.data(),
             scratch.up_out.data(), seq_len, ff_dim, embed_dim);

        // 10. SiLU(gate) * up
        cuda_kernels::silu_hadamard(scratch.gate_out.data(), scratch.up_out.data(),
                                     scratch.silu_out.data(), seq_len * ff_dim);

        // 11. Down projection
        gemm(scratch.silu_out.data(), lw.ff_down_w.data(),
             scratch.ffn_out.data(), seq_len, embed_dim, ff_dim);

        // 12. Residual: x += ffn_output
        cuda_kernels::add_inplace(scratch.x.data(), scratch.ffn_out.data(),
                                   seq_len * embed_dim);
    }

    kv_cache.len = seq_len;

    // --- Output head (last position only) ---
    const T* last_ptr = scratch.x.data()
                        + static_cast<size_t>(seq_len - 1) * embed_dim;
    cuda_kernels::rms_norm(last_ptr, weights.output_rms_gamma.data(),
                            scratch.last_row.data(), 1, embed_dim, rms_eps);

    gemm(scratch.last_row.data(), weights.output_proj_w.data(),
         scratch.logits.data(), 1, weights.output_proj_w.cols(), embed_dim);
}

// ============================================================
// GPU forward — single-token decode (cached)
// ============================================================

template<typename T>
void gpu_forward_cached(int token,
                        const GpuTransformerWeights<T>& weights,
                        GpuKVCache<T>& kv_cache,
                        GpuScratch<T>& scratch,
                        int num_heads, int num_kv_heads, int head_dim,
                        int embed_dim, int ff_dim, float rms_eps) {
    int kv_dim = num_kv_heads * head_dim;
    int heads_per_group = num_heads / num_kv_heads;
    int pos = kv_cache.len;

    // Upload single token → embedding lookup
    int* d_token;
    CUDA_CHECK(cudaMalloc(&d_token, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token, &token, sizeof(int), cudaMemcpyHostToDevice));
    cuda_kernels::embedding_lookup(d_token, weights.token_embedding.data(),
                                    scratch.x.data(), 1, embed_dim);
    CUDA_CHECK(cudaFree(d_token));

    int num_layers = static_cast<int>(weights.layers.size());
    for (int l = 0; l < num_layers; l++) {
        const auto& lw = weights.layers[l];

        // 1. Pre-norm
        cuda_kernels::rms_norm(scratch.x.data(), lw.rms1_gamma.data(),
                                scratch.norm_out.data(), 1, embed_dim, rms_eps);

        // 2. Q/K/V projections (Q pre-scaled by 1/sqrt(head_dim))
        float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        gemm(scratch.norm_out.data(), lw.W_q.data(),
             scratch.Q.data(), 1, embed_dim, embed_dim, attn_scale);
        gemm(scratch.norm_out.data(), lw.W_k.data(),
             scratch.K.data(), 1, kv_dim, embed_dim);
        gemm(scratch.norm_out.data(), lw.W_v.data(),
             scratch.V.data(), 1, kv_dim, embed_dim);

        // 3. RoPE at current position
        if (!lw.rope_cos.empty()) {
            cuda_kernels::rope(scratch.Q.data(), scratch.K.data(),
                                lw.rope_cos.data(), lw.rope_sin.data(),
                                1, num_heads, num_kv_heads, head_dim, pos);
        }

        // 4. Append K, V to cache at position `pos`
        kv_cache.layers[l].K.write_row(pos, scratch.K.data(), kv_dim);
        kv_cache.layers[l].V.write_row(pos, scratch.V.data(), kv_dim);
        int total_len = pos + 1;

        // 5. Per-head attention (Q already pre-scaled)
        T one = make_val<T>(1.0f);

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_group;

            const T* Q_h = scratch.Q.data() + h * head_dim;
            const T* K_h = kv_cache.layers[l].K.data() + kv_h * head_dim;
            const T* V_h = kv_cache.layers[l].V.data() + kv_h * head_dim;
            T* attn_h    = scratch.attn_out.data() + h * head_dim;

            // scores(1, total_len) = Q_h * K_h^T (already scaled)
            gemm_ABt(Q_h, embed_dim, K_h, kv_dim,
                     scratch.scores.data(), total_len,
                     1, total_len, head_dim);

            // Causal mask only (offset=pos → all positions visible)
            cuda_kernels::scale_causal_mask(
                scratch.scores.data(), 1, total_len, one, pos);

            // Softmax
            cuda_kernels::softmax(scratch.scores.data(), scratch.scores.data(),
                                   1, total_len);

            // attn_h(1, head_dim) = softmax * V_h
            gemm_strided(scratch.scores.data(), total_len,
                         V_h, kv_dim,
                         attn_h, embed_dim,
                         1, head_dim, total_len);
        }

        // 6. Output projection
        gemm(scratch.attn_out.data(), lw.W_o.data(),
             scratch.ffn_out.data(), 1, embed_dim, embed_dim);

        // 7. Residual
        cuda_kernels::add_inplace(scratch.x.data(), scratch.ffn_out.data(), embed_dim);

        // 8. Second pre-norm
        cuda_kernels::rms_norm(scratch.x.data(), lw.rms2_gamma.data(),
                                scratch.norm_out.data(), 1, embed_dim, rms_eps);

        // 9. Gated FFN
        gemm(scratch.norm_out.data(), lw.ff_gate_w.data(),
             scratch.gate_out.data(), 1, ff_dim, embed_dim);
        gemm(scratch.norm_out.data(), lw.ff_up_w.data(),
             scratch.up_out.data(), 1, ff_dim, embed_dim);

        // 10. SiLU(gate) * up
        cuda_kernels::silu_hadamard(scratch.gate_out.data(), scratch.up_out.data(),
                                     scratch.silu_out.data(), ff_dim);

        // 11. Down projection
        gemm(scratch.silu_out.data(), lw.ff_down_w.data(),
             scratch.ffn_out.data(), 1, embed_dim, ff_dim);

        // 12. Residual
        cuda_kernels::add_inplace(scratch.x.data(), scratch.ffn_out.data(), embed_dim);
    }

    kv_cache.len = pos + 1;

    // Output head
    cuda_kernels::rms_norm(scratch.x.data(), weights.output_rms_gamma.data(),
                            scratch.last_row.data(), 1, embed_dim, rms_eps);

    gemm(scratch.last_row.data(), weights.output_proj_w.data(),
         scratch.logits.data(), 1, weights.output_proj_w.cols(), embed_dim);
}

// ============================================================
// Transformer::generate_gpu — full generation loop on GPU
// ============================================================

template<typename T>
std::vector<int> Transformer<T>::generate_gpu(const std::vector<int>& prompt,
                                               int max_tokens) {
    TokenSampler<T> greedy;
    return generate_gpu(prompt, max_tokens, greedy);
}

template<typename T>
std::vector<int> Transformer<T>::generate_gpu(const std::vector<int>& prompt,
                                               int max_tokens,
                                               const TokenSampler<T>& sampler) {
    return generate_gpu(prompt, max_tokens, sampler, nullptr);
}

template<typename T>
std::vector<int> Transformer<T>::generate_gpu(const std::vector<int>& prompt,
                                               int max_tokens,
                                               const TokenSampler<T>& sampler,
                                               std::function<void(int)> on_token) {
    using GpuT = __half;

    prepare_gpu();

    auto& attn0 = blocks[0]->getAttention();
    int num_heads    = attn0.getNumHeads();
    int num_kv_heads = attn0.getNumKVHeads();
    int head_dim     = attn0.getHeadDim();
    int kv_dim       = attn0.getKVDim();
    int ff_dim       = blocks[0]->getFF1().getWeights().cols();
    float rms_eps    = static_cast<float>(output_rms->getEpsilon());

    gpu_kv_cache_->clear();

    std::vector<int> output = prompt;

    // Batched prefill
    gpu_forward_prefill(prompt, *gpu_weights_, *gpu_kv_cache_, *gpu_scratch_,
                         num_heads, num_kv_heads, head_dim, embed_dim, ff_dim, rms_eps);

    // Download logits as float for CPU sampling
    Matrix<T> logits = gpu_scratch_->logits.to_cpu_float();

    for (int t = 0; t < max_tokens; t++) {
        int token = sampler.sample(logits);
        output.push_back(token);
        if (on_token) on_token(token);

        if (std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end())
            break;
        if (gpu_kv_cache_->len >= max_seq_len) break;
        if (t == max_tokens - 1) break;

        // Single-token decode
        gpu_forward_cached(token, *gpu_weights_, *gpu_kv_cache_, *gpu_scratch_,
                            num_heads, num_kv_heads, head_dim, embed_dim, ff_dim, rms_eps);

        logits = gpu_scratch_->logits.to_cpu_float();
    }

    return output;
}

template<typename T>
void Transformer<T>::prepare_gpu() {
    using GpuT = __half;

    if (gpu_weights_) return;

    auto& attn0 = blocks[0]->getAttention();
    int kv_dim  = attn0.getKVDim();
    int ff_dim  = blocks[0]->getFF1().getWeights().cols();

    gpu_weights_ = std::make_unique<GpuTransformerWeights<GpuT>>(
        transfer_weights_to_gpu<GpuT>(*this));

    gpu_kv_cache_ = std::make_unique<GpuKVCache<GpuT>>();
    gpu_kv_cache_->layers.resize(blocks.size());
    for (auto& layer : gpu_kv_cache_->layers) {
        layer.K = CudaMatrix<GpuT>(max_seq_len, kv_dim);
        layer.V = CudaMatrix<GpuT>(max_seq_len, kv_dim);
    }

    gpu_scratch_ = std::make_unique<GpuScratch<GpuT>>();
    gpu_scratch_->allocate(max_seq_len, embed_dim, ff_dim, kv_dim, vocab_size);
}

// ============================================================
// Explicit instantiation (fp16 GPU inference)
// ============================================================

template void GpuScratch<__half>::allocate(int, int, int, int, int);

template GpuTransformerWeights<__half> transfer_weights_to_gpu(Transformer<float>&);

template void gpu_forward_prefill<__half>(
    const std::vector<int>&, const GpuTransformerWeights<__half>&,
    GpuKVCache<__half>&, GpuScratch<__half>&, int, int, int, int, int, float);

template void gpu_forward_cached<__half>(
    int, const GpuTransformerWeights<__half>&,
    GpuKVCache<__half>&, GpuScratch<__half>&, int, int, int, int, int, float);

template std::vector<int> Transformer<float>::generate_gpu(
    const std::vector<int>&, int);
template std::vector<int> Transformer<float>::generate_gpu(
    const std::vector<int>&, int, const TokenSampler<float>&);

#endif // ML_USE_CUDA
