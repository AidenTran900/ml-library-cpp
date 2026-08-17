#ifdef ML_USE_CUDA

#include "ml_lib/cuda/cuda-matrix.h"
#include "ml_lib/math/matrix.h"

// This was generated using claude

// Upload from CPU Matrix
template<typename T>
CudaMatrix<T>::CudaMatrix(const Matrix<T>& cpu)
    : m_rows(cpu.rows()), m_cols(cpu.cols())
{
    if (m_rows > 0 && m_cols > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, size_bytes()));
        CUDA_CHECK(cudaMemcpy(d_data, cpu.data(), size_bytes(), cudaMemcpyHostToDevice));
    }
}

// Download to CPU Matrix
template<typename T>
Matrix<T> CudaMatrix<T>::to_cpu() const {
    Matrix<T> result(m_rows, m_cols);
    if (d_data && m_rows > 0 && m_cols > 0) {
        CUDA_CHECK(cudaMemcpy(result.data(), d_data, size_bytes(), cudaMemcpyDeviceToHost));
    }
    return result;
}

// Row-major matmul via cuBLAS
// C(M,N) = A(M,K) * B(K,N)
// cuBLAS column-major: C^T = B^T * A^T
// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
template<>
CudaMatrix<float> CudaMatrix<float>::matmul(const CudaMatrix<float>& A, const CudaMatrix<float>& B) {
    int M = A.m_rows;
    int K = A.m_cols;
    int N = B.m_cols;

    CudaMatrix<float> C(M, N);
    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.d_data, N,
        A.d_data, K,
        &beta,
        C.d_data, N));

    return C;
}

// ============================================================
// from_cpu_float — upload float CPU data, storing as T on GPU
// ============================================================

template<>
CudaMatrix<float> CudaMatrix<float>::from_cpu_float(const Matrix<float>& cpu) {
    return CudaMatrix<float>(cpu);
}

template<>
CudaMatrix<__half> CudaMatrix<__half>::from_cpu_float(const Matrix<float>& cpu) {
    CudaMatrix<__half> result;
    result.m_rows = cpu.rows();
    result.m_cols = cpu.cols();
    if (result.m_rows > 0 && result.m_cols > 0) {
        size_t n = static_cast<size_t>(result.m_rows) * result.m_cols;
        std::vector<__half> temp(n);
        const float* src = cpu.data();
        for (size_t i = 0; i < n; i++) {
            temp[i] = __float2half(src[i]);
        }
        CUDA_CHECK(cudaMalloc(&result.d_data, result.size_bytes()));
        CUDA_CHECK(cudaMemcpy(result.d_data, temp.data(), result.size_bytes(),
                              cudaMemcpyHostToDevice));
    }
    return result;
}

// ============================================================
// to_cpu_float — download GPU data as float CPU Matrix
// ============================================================

template<>
Matrix<float> CudaMatrix<float>::to_cpu_float() const {
    return to_cpu();
}

template<>
Matrix<float> CudaMatrix<__half>::to_cpu_float() const {
    Matrix<float> result(m_rows, m_cols);
    if (d_data && m_rows > 0 && m_cols > 0) {
        size_t n = static_cast<size_t>(m_rows) * m_cols;
        std::vector<__half> temp(n);
        CUDA_CHECK(cudaMemcpy(temp.data(), d_data, size_bytes(), cudaMemcpyDeviceToHost));
        float* dst = result.data();
        for (size_t i = 0; i < n; i++) {
            dst[i] = __half2float(temp[i]);
        }
    }
    return result;
}

// ============================================================
// matmul for __half via cublasGemmEx (Tensor Core fp16)
// ============================================================

template<>
CudaMatrix<__half> CudaMatrix<__half>::matmul(const CudaMatrix<__half>& A,
                                               const CudaMatrix<__half>& B) {
    int M = A.m_rows;
    int K = A.m_cols;
    int N = B.m_cols;

    CudaMatrix<__half> C(M, N);
    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.d_data, CUDA_R_16F, N,
        A.d_data, CUDA_R_16F, K,
        &beta,
        C.d_data, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    return C;
}

// Explicit instantiation
// CudaMatrix<float>: full class instantiation (all methods)
template class CudaMatrix<float>;
// CudaMatrix<__half>: only header-inline + explicit specializations above
// (no blanket instantiation — Matrix<__half> doesn't exist for to_cpu())

#endif // ML_USE_CUDA
