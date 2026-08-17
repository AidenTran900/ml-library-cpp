#ifdef ML_USE_CUDA

#include "ml_lib/cuda/cuda-kernels.h"
#include <cfloat>
#include <cmath>


// This was generated using claude

// ============================================================
// Type conversion helpers — load/store in T, compute in float
// ============================================================

template<typename T> __host__ __device__ __forceinline__ float to_float(T x);
template<> __host__ __device__ __forceinline__ float to_float<float>(float x) { return x; }
template<> __host__ __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }

template<typename T> __host__ __device__ __forceinline__ T from_float(float x);
template<> __host__ __device__ __forceinline__ float from_float<float>(float x) { return x; }
template<> __host__ __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// ============================================================
// Kernel 1: RMS Norm
// output[i,j] = (input[i,j] / rms_i) * gamma[j]
// One block per row, shared memory reduction for sum-of-squares
// Accumulation always in float for numerical stability
// ============================================================

template<typename T>
__global__ void rmsnorm_kernel(const T* input, const T* gamma, T* output,
                                int cols, float epsilon) {
    int row = blockIdx.x;
    const T* x = input + row * cols;
    T* o = output + row * cols;

    extern __shared__ char shared_mem[];
    float* sdata = reinterpret_cast<float*>(shared_mem);

    // Compute sum of squares in float
    float thread_sum = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(x[j]);
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / cols + epsilon);

    // Normalize and scale
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float val = to_float(x[j]);
        float g = to_float(gamma[j]);
        o[j] = from_float<T>((val / rms) * g);
    }
}

template<typename T>
void cuda_kernels::rms_norm(const T* input, const T* gamma, T* output,
                            int rows, int cols, float epsilon, cudaStream_t stream) {
    int threads = min(256, cols);
    int shared_size = threads * sizeof(float);
    rmsnorm_kernel<<<rows, threads, shared_size, stream>>>(input, gamma, output, cols, epsilon);
}

// ============================================================
// Kernel 2: RoPE (Rotary Positional Encoding)
// Apply rotation to Q and K in-place
// Grid: (seq_len, num_heads + num_kv_heads)
// Block: head_dim / 2
// ============================================================

template<typename T>
__global__ void rope_kernel(T* Q, T* K,
                            const T* cos_table, const T* sin_table,
                            int num_heads, int num_kv_heads,
                            int head_dim, int embed_dim, int kv_dim,
                            int start_pos) {
    int pos = blockIdx.x + start_pos;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x;  // which pair (0..head_dim/2-1)

    int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    float cos_val = to_float(cos_table[pos * half_dim + pair_idx]);
    float sin_val = to_float(sin_table[pos * half_dim + pair_idx]);

    int seq_idx = blockIdx.x;  // position within the current input

    if (head_idx < num_heads) {
        // Rotate Q head
        int offset = seq_idx * embed_dim + head_idx * head_dim + pair_idx * 2;
        float x0 = to_float(Q[offset]);
        float x1 = to_float(Q[offset + 1]);
        Q[offset]     = from_float<T>(x0 * cos_val - x1 * sin_val);
        Q[offset + 1] = from_float<T>(x0 * sin_val + x1 * cos_val);
    } else {
        // Rotate K head
        int kv_head = head_idx - num_heads;
        int offset = seq_idx * kv_dim + kv_head * head_dim + pair_idx * 2;
        float x0 = to_float(K[offset]);
        float x1 = to_float(K[offset + 1]);
        K[offset]     = from_float<T>(x0 * cos_val - x1 * sin_val);
        K[offset + 1] = from_float<T>(x0 * sin_val + x1 * cos_val);
    }
}

template<typename T>
void cuda_kernels::rope(T* Q, T* K, const T* cos_table, const T* sin_table,
                        int seq_len, int num_heads, int num_kv_heads,
                        int head_dim, int start_pos, cudaStream_t stream) {
    int embed_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    dim3 grid(seq_len, num_heads + num_kv_heads);
    int threads = head_dim / 2;
    rope_kernel<<<grid, threads, 0, stream>>>(Q, K, cos_table, sin_table,
                                               num_heads, num_kv_heads,
                                               head_dim, embed_dim, kv_dim,
                                               start_pos);
}

// ============================================================
// Kernel 3: Row-wise Softmax
// One block per row, shared mem for max and sum reduction
// Accumulation in float for numerical stability
// ============================================================

template<typename T>
__global__ void softmax_kernel(const T* input, T* output, int cols) {
    int row = blockIdx.x;
    const T* x = input + row * cols;
    T* o = output + row * cols;

    extern __shared__ char shared_mem[];
    float* sdata = reinterpret_cast<float*>(shared_mem);

    // Find max
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        thread_max = fmaxf(thread_max, to_float(x[j]));
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Compute sum of exp (no intermediate fp16 storage — avoids rounding errors)
    float thread_sum = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        thread_sum += expf(to_float(x[j]) - max_val);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();

    // Recompute exp and normalize in one pass (avoids fp16 intermediate rounding)
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        o[j] = from_float<T>(expf(to_float(x[j]) - max_val) / sum_val);
    }
}

template<typename T>
void cuda_kernels::softmax(const T* input, T* output, int rows, int cols, cudaStream_t stream) {
    int threads = min(256, cols);
    // Round up to next power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int shared_size = threads * sizeof(float);
    softmax_kernel<<<rows, threads, shared_size, stream>>>(input, output, cols);
}

// ============================================================
// Kernel 4: SiLU + Hadamard
// output[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
// ============================================================

template<typename T>
__global__ void silu_hadamard_kernel(const T* gate, const T* up, T* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gate[idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = from_float<T>(silu_g * to_float(up[idx]));
    }
}

template<typename T>
void cuda_kernels::silu_hadamard(const T* gate, const T* up, T* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_hadamard_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, n);
}

// ============================================================
// Kernel 5: In-place Add (residual connection)
// a[i] += b[i]
// ============================================================

template<typename T>
__global__ void add_inplace_kernel(T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = from_float<T>(to_float(a[idx]) + to_float(b[idx]));
    }
}

template<typename T>
void cuda_kernels::add_inplace(T* a, const T* b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads, 0, stream>>>(a, b, n);
}

// ============================================================
// Kernel 6: Scale + Causal Mask
// scores[i,j] = (j <= i + offset) ? scores[i,j] * scale : -inf
// ============================================================

template<typename T>
__global__ void scale_causal_mask_kernel(T* scores, int rows, int cols, float scale, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        if (j <= i + offset) {
            scores[idx] = from_float<T>(to_float(scores[idx]) * scale);
        } else {
            scores[idx] = from_float<T>(-FLT_MAX);
        }
    }
}

template<typename T>
void cuda_kernels::scale_causal_mask(T* scores, int rows, int cols, T scale, int offset, cudaStream_t stream) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float scale_f = to_float(scale);
    scale_causal_mask_kernel<<<blocks, threads, 0, stream>>>(scores, rows, cols, scale_f, offset);
}

// ============================================================
// Kernel 7: Embedding Lookup
// output[i,:] = table[token_ids[i], :]
// ============================================================

template<typename T>
__global__ void embedding_lookup_kernel(const int* token_ids, const T* table,
                                         T* output, int embed_dim) {
    int token_idx = blockIdx.x;
    int token_id = token_ids[token_idx];
    const T* src = table + static_cast<size_t>(token_id) * embed_dim;
    T* dst = output + static_cast<size_t>(token_idx) * embed_dim;

    for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
        dst[j] = src[j];
    }
}

template<typename T>
void cuda_kernels::embedding_lookup(const int* token_ids, const T* table, T* output,
                                    int num_tokens, int embed_dim, cudaStream_t stream) {
    int threads = min(256, embed_dim);
    embedding_lookup_kernel<<<num_tokens, threads, 0, stream>>>(token_ids, table, output, embed_dim);
}

// Explicit instantiation — float
template void cuda_kernels::rms_norm<float>(const float*, const float*, float*, int, int, float, cudaStream_t);
template void cuda_kernels::rope<float>(float*, float*, const float*, const float*, int, int, int, int, int, cudaStream_t);
template void cuda_kernels::softmax<float>(const float*, float*, int, int, cudaStream_t);
template void cuda_kernels::silu_hadamard<float>(const float*, const float*, float*, int, cudaStream_t);
template void cuda_kernels::add_inplace<float>(float*, const float*, int, cudaStream_t);
template void cuda_kernels::scale_causal_mask<float>(float*, int, int, float, int, cudaStream_t);
template void cuda_kernels::embedding_lookup<float>(const int*, const float*, float*, int, int, cudaStream_t);

// Explicit instantiation — __half
template void cuda_kernels::rms_norm<__half>(const __half*, const __half*, __half*, int, int, float, cudaStream_t);
template void cuda_kernels::rope<__half>(__half*, __half*, const __half*, const __half*, int, int, int, int, int, cudaStream_t);
template void cuda_kernels::softmax<__half>(const __half*, __half*, int, int, cudaStream_t);
template void cuda_kernels::silu_hadamard<__half>(const __half*, const __half*, __half*, int, cudaStream_t);
template void cuda_kernels::add_inplace<__half>(__half*, const __half*, int, cudaStream_t);
template void cuda_kernels::scale_causal_mask<__half>(__half*, int, int, __half, int, cudaStream_t);
template void cuda_kernels::embedding_lookup<__half>(const int*, const __half*, __half*, int, int, cudaStream_t);

#endif // ML_USE_CUDA
