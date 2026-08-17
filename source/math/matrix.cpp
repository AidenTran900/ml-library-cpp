#include "ml_lib/math/matrix.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>
#if ML_HAS_AVX2 && ML_USE_SIMD
    #include <immintrin.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif

const int DEFAULT_BLOCK_SIZE = 64;
const int PACK_STRIDE = DEFAULT_BLOCK_SIZE + 16;

template<typename T>
Matrix<T>::Matrix()
    : m_rows(0),
      m_cols(0),
      m_data() {}

template<typename T>
Matrix<T>::Matrix(int rows, int cols, double init_val)
    : m_rows(rows),
      m_cols(cols),
      m_data(static_cast<size_t>(rows) * static_cast<size_t>(cols), static_cast<T>(init_val)) {}

template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& vec) {
    if (vec.empty()) {
        m_rows = 0;
        m_cols = 0;
        return;
    }
    m_rows = static_cast<int>(vec.size());
    m_cols = static_cast<int>(vec[0].size());
    m_data.resize(static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols));
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            m_data[static_cast<size_t>(i * m_cols + j)] = vec[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
}

template<typename T>
T& Matrix<T>::operator()(int i, int j) { // Setter
    return m_data[static_cast<size_t>(i * m_cols + j)];
}

template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    return m_data[static_cast<size_t>(i * m_cols + j)];
}

template<typename T>
const T* Matrix<T>::getRow(int row) const {
    return &m_data[static_cast<size_t>(row * m_cols)];
}

template<typename T>
T* Matrix<T>::getRow(int row) {
    return &m_data[static_cast<size_t>(row * m_cols)];
}

template<typename T>
std::vector<T> Matrix<T>::getRowVector(int row) const {
    const T* row_ptr = getRow(row);
    return std::vector<T>(row_ptr, row_ptr + m_cols);
}

template<typename T>
Matrix<T> Matrix<T>::row(int r) const {
    Matrix result(1, m_cols);
    for (int j = 0; j < m_cols; j++) {
        result(0, j) = (*this)(r, j);
    }
    return result;
}

template<typename T>
void Matrix<T>::swapRows(int row1, int row2) {
    if (row1 == row2) return;
    T* r1 = getRow(row1);
    T* r2 = getRow(row2);
    for (int j = 0; j < m_cols; j++) {
        std::swap(r1[j], r2[j]);
    }
}

template<typename T>
void Matrix<T>::print() const {
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            std::cout << std::setw(8) << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}


template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }
    Matrix result(m_rows, m_cols);
    size_t i = 0;
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);

    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= size; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
            __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
            __m256d sum = _mm256_add_pd(vec1, vec2);
            _mm256_storeu_pd(&result.m_data[i], sum);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= size; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(&m_data[i]);
            __m256 vec2 = _mm256_loadu_ps(&other.m_data[i]);
            __m256 sum = _mm256_add_ps(vec1, vec2);
            _mm256_storeu_ps(&result.m_data[i], sum);
        }
    }
    #endif

    for (; i < size; i++) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    return result;
}


template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (m_cols != other.m_cols || m_rows != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible.");
    }
    Matrix result(m_rows, m_cols);
    size_t i = 0;
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);

    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= size; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
            __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
            __m256d diff = _mm256_sub_pd(vec1, vec2);
            _mm256_storeu_pd(&result.m_data[i], diff);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= size; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(&m_data[i]);
            __m256 vec2 = _mm256_loadu_ps(&other.m_data[i]);
            __m256 diff = _mm256_sub_ps(vec1, vec2);
            _mm256_storeu_ps(&result.m_data[i], diff);
        }
    }
    #endif

    for (; i < size; i++) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (m_cols != other.m_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible. "
            "Matrix 1 cols (" + std::to_string(m_cols) +
            ") != Matrix 2 rows (" + std::to_string(other.m_rows) + ").");
    }

    Matrix result(m_rows, other.m_cols);
    const int M = m_rows;
    const int N = other.m_cols;
    const int K = m_cols;
    T* __restrict__ C = result.m_data.data();
    const T* __restrict__ A = m_data.data();
    const T* __restrict__ B = other.m_data.data();


    if (M == 1) {
        #if ML_HAS_AVX2 && ML_USE_SIMD
        if constexpr (std::is_same_v<T, double>) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                int j = jj;
                for (; j + 16 <= j_end; j += 16) {
                    __m256d c0 = _mm256_loadu_pd(&C[j]);
                    __m256d c1 = _mm256_loadu_pd(&C[j + 4]);
                    __m256d c2 = _mm256_loadu_pd(&C[j + 8]);
                    __m256d c3 = _mm256_loadu_pd(&C[j + 12]);
                    for (int k = 0; k < K; k++) {
                        __m256d a_k = _mm256_set1_pd(A[k]);
                        const T* b_row = &B[static_cast<size_t>(k * N + j)];
                        c0 = _mm256_fmadd_pd(a_k, _mm256_loadu_pd(b_row), c0);
                        c1 = _mm256_fmadd_pd(a_k, _mm256_loadu_pd(b_row + 4), c1);
                        c2 = _mm256_fmadd_pd(a_k, _mm256_loadu_pd(b_row + 8), c2);
                        c3 = _mm256_fmadd_pd(a_k, _mm256_loadu_pd(b_row + 12), c3);
                    }
                    _mm256_storeu_pd(&C[j], c0);
                    _mm256_storeu_pd(&C[j + 4], c1);
                    _mm256_storeu_pd(&C[j + 8], c2);
                    _mm256_storeu_pd(&C[j + 12], c3);
                }
                for (; j + 4 <= j_end; j += 4) {
                    __m256d c_vec = _mm256_loadu_pd(&C[j]);
                    for (int k = 0; k < K; k++) {
                        __m256d a_k = _mm256_set1_pd(A[k]);
                        __m256d b_vec = _mm256_loadu_pd(&B[static_cast<size_t>(k * N + j)]);
                        c_vec = _mm256_fmadd_pd(a_k, b_vec, c_vec);
                    }
                    _mm256_storeu_pd(&C[j], c_vec);
                }
                for (; j < j_end; j++) {
                    T sum = C[j];
                    for (int k = 0; k < K; k++) {
                        sum += A[k] * B[static_cast<size_t>(k * N + j)];
                    }
                    C[j] = sum;
                }
            }
        } else if constexpr (std::is_same_v<T, float>) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                int j = jj;
                for (; j + 32 <= j_end; j += 32) {
                    __m256 c0 = _mm256_loadu_ps(&C[j]);
                    __m256 c1 = _mm256_loadu_ps(&C[j + 8]);
                    __m256 c2 = _mm256_loadu_ps(&C[j + 16]);
                    __m256 c3 = _mm256_loadu_ps(&C[j + 24]);
                    for (int k = 0; k < K; k++) {
                        __m256 a_k = _mm256_set1_ps(A[k]);
                        const T* b_row = &B[static_cast<size_t>(k * N + j)];
                        c0 = _mm256_fmadd_ps(a_k, _mm256_loadu_ps(b_row), c0);
                        c1 = _mm256_fmadd_ps(a_k, _mm256_loadu_ps(b_row + 8), c1);
                        c2 = _mm256_fmadd_ps(a_k, _mm256_loadu_ps(b_row + 16), c2);
                        c3 = _mm256_fmadd_ps(a_k, _mm256_loadu_ps(b_row + 24), c3);
                    }
                    _mm256_storeu_ps(&C[j], c0);
                    _mm256_storeu_ps(&C[j + 8], c1);
                    _mm256_storeu_ps(&C[j + 16], c2);
                    _mm256_storeu_ps(&C[j + 24], c3);
                }
                for (; j + 8 <= j_end; j += 8) {
                    __m256 c_vec = _mm256_loadu_ps(&C[j]);
                    for (int k = 0; k < K; k++) {
                        __m256 a_k = _mm256_set1_ps(A[k]);
                        __m256 b_vec = _mm256_loadu_ps(&B[static_cast<size_t>(k * N + j)]);
                        c_vec = _mm256_fmadd_ps(a_k, b_vec, c_vec);
                    }
                    _mm256_storeu_ps(&C[j], c_vec);
                }
                for (; j < j_end; j++) {
                    T sum = C[j];
                    for (int k = 0; k < K; k++) {
                        sum += A[k] * B[static_cast<size_t>(k * N + j)];
                    }
                    C[j] = sum;
                }
            }
        } else {
        #else
        {
        #endif
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                for (int k = 0; k < K; k++) {
                    const T a_k = A[k];
                    for (int j = jj; j < j_end; j++) {
                        C[j] += a_k * B[static_cast<size_t>(k * N + j)];
                    }
                }
            }
        }
        return result;
    }

    // general matrix multiply (M > 1)
    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int ii = 0; ii < M; ii += DEFAULT_BLOCK_SIZE) {
            const int i_end = std::min(ii + DEFAULT_BLOCK_SIZE, M);
            for (int kk = 0; kk < K; kk += DEFAULT_BLOCK_SIZE) {
                const int k_end = std::min(kk + DEFAULT_BLOCK_SIZE, K);
                const int kb = k_end - kk;
                for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                    const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                    const int jb = j_end - jj;

                    T b_pack[DEFAULT_BLOCK_SIZE * PACK_STRIDE];
                    for (int kl = 0; kl < kb; kl++) {
                        std::memcpy(&b_pack[static_cast<size_t>(kl) * PACK_STRIDE],
                                    &B[static_cast<size_t>((kk + kl) * N + jj)],
                                    static_cast<size_t>(jb) * sizeof(T));
                    }

                    for (int i = ii; i < i_end; i++) {
                        int j = jj;
                        for (; j + 16 <= j_end; j += 16) {
                            size_t base = static_cast<size_t>(i * N + j);
                            __m256d c0 = _mm256_loadu_pd(&C[base]);
                            __m256d c1 = _mm256_loadu_pd(&C[base + 4]);
                            __m256d c2 = _mm256_loadu_pd(&C[base + 8]);
                            __m256d c3 = _mm256_loadu_pd(&C[base + 12]);
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                __m256d a_ik = _mm256_set1_pd(A[static_cast<size_t>(i * K + kk + kl)]);
                                const T* b_row = &b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0];
                                c0 = _mm256_fmadd_pd(a_ik, _mm256_loadu_pd(b_row), c0);
                                c1 = _mm256_fmadd_pd(a_ik, _mm256_loadu_pd(b_row + 4), c1);
                                c2 = _mm256_fmadd_pd(a_ik, _mm256_loadu_pd(b_row + 8), c2);
                                c3 = _mm256_fmadd_pd(a_ik, _mm256_loadu_pd(b_row + 12), c3);
                            }
                            _mm256_storeu_pd(&C[base], c0);
                            _mm256_storeu_pd(&C[base + 4], c1);
                            _mm256_storeu_pd(&C[base + 8], c2);
                            _mm256_storeu_pd(&C[base + 12], c3);
                        }
                        for (; j + 4 <= j_end; j += 4) {
                            size_t c_idx = static_cast<size_t>(i * N + j);
                            __m256d c_vec = _mm256_loadu_pd(&C[c_idx]);
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                __m256d a_ik = _mm256_set1_pd(A[static_cast<size_t>(i * K + kk + kl)]);
                                __m256d b_vec = _mm256_loadu_pd(&b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0]);
                                c_vec = _mm256_fmadd_pd(a_ik, b_vec, c_vec);
                            }
                            _mm256_storeu_pd(&C[c_idx], c_vec);
                        }
                        for (; j < j_end; j++) {
                            size_t c_idx = static_cast<size_t>(i * N + j);
                            T sum = C[c_idx];
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                sum += A[static_cast<size_t>(i * K + kk + kl)] * b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0];
                            }
                            C[c_idx] = sum;
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, float>) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int ii = 0; ii < M; ii += DEFAULT_BLOCK_SIZE) {
            const int i_end = std::min(ii + DEFAULT_BLOCK_SIZE, M);
            for (int kk = 0; kk < K; kk += DEFAULT_BLOCK_SIZE) {
                const int k_end = std::min(kk + DEFAULT_BLOCK_SIZE, K);
                const int kb = k_end - kk;
                for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                    const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                    const int jb = j_end - jj;

                    T b_pack[DEFAULT_BLOCK_SIZE * PACK_STRIDE];
                    for (int kl = 0; kl < kb; kl++) {
                        std::memcpy(&b_pack[static_cast<size_t>(kl) * PACK_STRIDE],
                                    &B[static_cast<size_t>((kk + kl) * N + jj)],
                                    static_cast<size_t>(jb) * sizeof(T));
                    }

                    for (int i = ii; i < i_end; i++) {
                        int j = jj;
                        for (; j + 32 <= j_end; j += 32) {
                            size_t base = static_cast<size_t>(i * N + j);
                            __m256 c0 = _mm256_loadu_ps(&C[base]);
                            __m256 c1 = _mm256_loadu_ps(&C[base + 8]);
                            __m256 c2 = _mm256_loadu_ps(&C[base + 16]);
                            __m256 c3 = _mm256_loadu_ps(&C[base + 24]);
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                __m256 a_ik = _mm256_set1_ps(A[static_cast<size_t>(i * K + kk + kl)]);
                                const T* b_row = &b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0];
                                c0 = _mm256_fmadd_ps(a_ik, _mm256_loadu_ps(b_row), c0);
                                c1 = _mm256_fmadd_ps(a_ik, _mm256_loadu_ps(b_row + 8), c1);
                                c2 = _mm256_fmadd_ps(a_ik, _mm256_loadu_ps(b_row + 16), c2);
                                c3 = _mm256_fmadd_ps(a_ik, _mm256_loadu_ps(b_row + 24), c3);
                            }
                            _mm256_storeu_ps(&C[base], c0);
                            _mm256_storeu_ps(&C[base + 8], c1);
                            _mm256_storeu_ps(&C[base + 16], c2);
                            _mm256_storeu_ps(&C[base + 24], c3);
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            size_t c_idx = static_cast<size_t>(i * N + j);
                            __m256 c_vec = _mm256_loadu_ps(&C[c_idx]);
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                __m256 a_ik = _mm256_set1_ps(A[static_cast<size_t>(i * K + kk + kl)]);
                                __m256 b_vec = _mm256_loadu_ps(&b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0]);
                                c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);
                            }
                            _mm256_storeu_ps(&C[c_idx], c_vec);
                        }
                        for (; j < j_end; j++) {
                            size_t c_idx = static_cast<size_t>(i * N + j);
                            T sum = C[c_idx];
                            const int jl0 = j - jj;
                            for (int kl = 0; kl < kb; kl++) {
                                sum += A[static_cast<size_t>(i * K + kk + kl)] * b_pack[static_cast<size_t>(kl) * PACK_STRIDE + jl0];
                            }
                            C[c_idx] = sum;
                        }
                    }
                }
            }
        }
    } else {
    #else
    {
    #endif
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int ii = 0; ii < M; ii += DEFAULT_BLOCK_SIZE) {
            const int i_end = std::min(ii + DEFAULT_BLOCK_SIZE, M);
            for (int kk = 0; kk < K; kk += DEFAULT_BLOCK_SIZE) {
                const int k_end = std::min(kk + DEFAULT_BLOCK_SIZE, K);
                for (int jj = 0; jj < N; jj += DEFAULT_BLOCK_SIZE) {
                    const int j_end = std::min(jj + DEFAULT_BLOCK_SIZE, N);
                    for (int i = ii; i < i_end; i++) {
                        for (int k = kk; k < k_end; k++) {
                            const T a_ik = A[static_cast<size_t>(i * K + k)];
                            for (int j = jj; j < j_end; j++) {
                                C[static_cast<size_t>(i * N + j)] += a_ik * B[static_cast<size_t>(k * N + j)];
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}


template<typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }

    size_t i = 0;
    Matrix result(m_rows, m_cols);
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);

    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        for (; i + 4 <= size; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
            __m256d vec2 = _mm256_loadu_pd(&other.m_data[i]);
            __m256d product = _mm256_mul_pd(vec1, vec2);
            _mm256_storeu_pd(&result.m_data[i], product);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        for (; i + 8 <= size; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(&m_data[i]);
            __m256 vec2 = _mm256_loadu_ps(&other.m_data[i]);
            __m256 product = _mm256_mul_ps(vec1, vec2);
            _mm256_storeu_ps(&result.m_data[i], product);
        }
    }
    #endif

    for (; i < size; i++) {
        result.m_data[i] = m_data[i] * other.m_data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(double scalar) const {
    Matrix result(m_rows, m_cols);
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
    size_t i = 0;

    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        __m256d scalar_vec = _mm256_set1_pd(scalar);
        for (; i + 4 <= size; i += 4) {
            __m256d vec = _mm256_loadu_pd(&m_data[i]);
            __m256d scaled = _mm256_mul_pd(vec, scalar_vec);
            _mm256_storeu_pd(&result.m_data[i], scaled);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        __m256 scalar_vec = _mm256_set1_ps(static_cast<float>(scalar));
        for (; i + 8 <= size; i += 8) {
            __m256 vec = _mm256_loadu_ps(&m_data[i]);
            __m256 scaled = _mm256_mul_ps(vec, scalar_vec);
            _mm256_storeu_ps(&result.m_data[i], scaled);
        }
    }
    #endif

    T scalar_t = static_cast<T>(scalar);
    for (; i < size; i++) {
        result.m_data[i] = m_data[i] * scalar_t;
    }
    return result;
}

template<typename T>
EliminationResult<T> Matrix<T>::forwardElimination(const Matrix& m, const Matrix& aug) {
    if (m.empty()) {
        return {m, aug, 0};
    }

    bool is_augmented = !aug.empty();
    if (is_augmented && m.m_rows != aug.m_rows) {
        throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    int pivot_row = 0;
    int swaps = 0;

    for (int j = 0; j < m_c.m_cols && pivot_row < m_c.m_rows; j++) {
        int max_row_ind = pivot_row;
        double max_val = std::abs(static_cast<double>(m_c(pivot_row, j)));

        for (int i = pivot_row + 1; i < m_c.m_rows; i++) {
            double abs_val = std::abs(static_cast<double>(m_c(i, j)));
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row_ind = i;
            }
        }

        if (max_row_ind != pivot_row) {
            m_c.swapRows(pivot_row, max_row_ind);
            if (is_augmented) {
                aug_c.swapRows(pivot_row, max_row_ind);
            }
            swaps++;
        }

        if (std::abs(static_cast<double>(m_c(pivot_row, j))) < 1e-9) {
            m_c(pivot_row, j) = static_cast<T>(0.0);
            continue;
        }

        T pivot = m_c(pivot_row, j);

        for (int i = pivot_row + 1; i < m_c.m_rows; i++) {
            T target = m_c(i, j);
            T c = target / pivot;

            T* current_row = m_c.getRow(i);
            const T* pivot_row_data = m_c.getRow(pivot_row);
            for (int z = j; z < m_c.m_cols; z++) {
                current_row[z] -= pivot_row_data[z] * c;
            }

            if (is_augmented) {
                T* aug_current = aug_c.getRow(i);
                const T* aug_pivot = aug_c.getRow(pivot_row);
                for (int z = 0; z < aug_c.m_cols; z++) {
                    aug_current[z] -= aug_pivot[z] * c;
                }
            }

            current_row[j] = static_cast<T>(0.0);
        }

        pivot_row++;
    }

    return {m_c, aug_c, swaps};
}

template<typename T>
EliminationResult<T> Matrix<T>::backwardElimination(const Matrix& m, const Matrix& aug) {
    if (m.empty()) {
        return {m, aug, 0};
    }

    bool is_augmented = !aug.empty();
    if (is_augmented && m.m_rows != aug.m_rows) {
        throw std::invalid_argument("Augmented matrix row count must match input matrix row count.");
    }

    Matrix m_c = m;
    Matrix aug_c = aug;

    for (int i = m_c.m_rows - 1; i >= 0; i--) {

        int pivot_col = -1;
        for (int j = 0; j < m_c.m_cols; j++) {
            if (std::abs(static_cast<double>(m_c(i, j))) > 1e-9) {
                pivot_col = j;
                break;
            }
        }

        if (pivot_col == -1) {
            continue;
        }

        T pivot_val = m_c(i, pivot_col);

        T* pivot_row_data = m_c.getRow(i);
        for (int j = pivot_col; j < m_c.m_cols; j++) {
            pivot_row_data[j] /= pivot_val;
        }
        if (is_augmented) {
            T* aug_pivot_row = aug_c.getRow(i);
            for (int j = 0; j < aug_c.m_cols; j++) {
                aug_pivot_row[j] /= pivot_val;
            }
        }
        pivot_row_data[pivot_col] = static_cast<T>(1.0);

        for (int k = i - 1; k >= 0; k--) {
            T target_val = m_c(k, pivot_col);

            T* target_row = m_c.getRow(k);
            for (int j = pivot_col; j < m_c.m_cols; j++) {
                target_row[j] -= target_val * pivot_row_data[j];
            }

            if (is_augmented) {
                T* aug_target = aug_c.getRow(k);
                const T* aug_pivot = aug_c.getRow(i);
                for (int j = 0; j < aug_c.m_cols; j++) {
                    aug_target[j] -= target_val * aug_pivot[j];
                }
            }
            target_row[pivot_col] = static_cast<T>(0.0);
        }
    }

    return {m_c, aug_c, 0};
}

template<typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (empty()) {
        throw std::invalid_argument("Cannot invert an empty matrix.");
    }

    if (m_rows != m_cols) {
        throw std::invalid_argument("Cannot invert a non-square matrix.");
    }

    Matrix identity(m_rows, m_rows);
    for (int i = 0; i < m_rows; i++) {
        identity(i, i) = static_cast<T>(1.0);
    }

    EliminationResult<T> forward_result = forwardElimination(*this, identity);

    for (int i = 0; i < m_rows; i++) {
        if (std::abs(static_cast<double>(forward_result.matrix(i, i))) < 1e-9) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
    }

    EliminationResult<T> backward_result = backwardElimination(forward_result.matrix, forward_result.augmented);

    return backward_result.augmented;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix result(m_cols, m_rows);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::verticalConcat(const Matrix& other) const {
    if (empty()) return other;
    if (other.empty()) return *this;
    if (m_cols != other.m_cols) {
        throw std::invalid_argument("Column count must match for vertical concatenation.");
    }
    Matrix result(m_rows + other.m_rows, m_cols);
    for (int i = 0; i < m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(i, j) = (*this)(i, j);
        }
    }
    for (int i = 0; i < other.m_rows; i++) {
        for (int j = 0; j < m_cols; j++) {
            result(m_rows + i, j) = other(i, j);
        }
    }
    return result;
}

template<typename T>
void Matrix<T>::appendRow(const Matrix& row) {
    if (row.m_rows != 1) {
        throw std::invalid_argument("appendRow expects a single-row matrix.");
    }
    if (empty()) {
        m_cols = row.m_cols;
        m_rows = 0;
    }
    if (m_cols != row.m_cols) {
        throw std::invalid_argument("Column count must match for appendRow.");
    }
    m_data.insert(m_data.end(), row.m_data.begin(), row.m_data.end());
    m_rows++;
}

template<typename T>
T Matrix<T>::determinant() const {
    if (empty()) {
        return static_cast<T>(0.0);
    }

    if (m_rows != m_cols) {
        throw std::invalid_argument("Matrix dimensions are not square.");
    }
    if (m_rows == 1 && m_cols == 1) {
        return (*this)(0, 0);
    }

    if (m_rows == 2 && m_cols == 2) {
        return (*this)(0, 0) * (*this)(1, 1) - (*this)(1, 0) * (*this)(0, 1);
    }

    EliminationResult<T> elim_result = forwardElimination(*this);

    T det = static_cast<T>(1.0);
    for (int i = 0; i < m_rows; i++) {
        if (std::abs(static_cast<double>(elim_result.matrix(i, i))) < 1e-9) {
            return static_cast<T>(0.0);
        }
        det *= elim_result.matrix(i, i);
    }

    if (elim_result.swaps % 2 != 0) {
        det *= static_cast<T>(-1.0);
    }

    return det;
}

template<typename T>
T Matrix<T>::dot(const Matrix& m) const {
    if (rows() != m.rows() || cols() != m.cols()) {
        throw std::invalid_argument("Dimension mismatch for dot product");
    }

    T result = static_cast<T>(0.0);
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
    size_t i = 0;

    #if ML_HAS_AVX2 && ML_USE_SIMD
    if constexpr (std::is_same_v<T, double>) {
        __m256d sum_vec = _mm256_setzero_pd();

        for (; i + 4 <= size; i += 4) {
            __m256d vec1 = _mm256_loadu_pd(&m_data[i]);
            __m256d vec2 = _mm256_loadu_pd(&m.m_data[i]);
            sum_vec = _mm256_fmadd_pd(vec1, vec2, sum_vec);
        }

        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum_vec);
        result = static_cast<T>(temp[0] + temp[1] + temp[2] + temp[3]);
    } else if constexpr (std::is_same_v<T, float>) {
        __m256 sum_vec = _mm256_setzero_ps();

        for (; i + 8 <= size; i += 8) {
            __m256 vec1 = _mm256_loadu_ps(&m_data[i]);
            __m256 vec2 = _mm256_loadu_ps(&m.m_data[i]);
            sum_vec = _mm256_fmadd_ps(vec1, vec2, sum_vec);
        }

        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum_vec);
        result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
    #endif

    for (; i < size; i++) {
        result += m_data[i] * m.m_data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sign() const {
    Matrix result(m_rows, m_cols);
    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
    for (size_t i = 0; i < size; i++) {
        T val = m_data[i];
        result.m_data[i] = (val > static_cast<T>(0.0)) ? static_cast<T>(1.0) : ((val < static_cast<T>(0.0)) ? static_cast<T>(-1.0) : static_cast<T>(0.0));
    }
    return result;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }

    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
    for (size_t i = 0; i < size; i++) {
        if (std::abs(static_cast<double>(m_data[i]) - static_cast<double>(other.m_data[i])) > 1e-15) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix& other) const {
    return !(*this == other);
}

template<typename T>
bool Matrix<T>::approxEqual(const Matrix& other, double epsilon) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }

    size_t size = static_cast<size_t>(m_rows) * static_cast<size_t>(m_cols);
    for (size_t i = 0; i < size; i++) {
        if (std::abs(static_cast<double>(m_data[i]) - static_cast<double>(other.m_data[i])) > epsilon) {
            return false;
        }
    }
    return true;
}

// Explicit template instantiation
template class Matrix<float>;
template class Matrix<double>;
template struct EliminationResult<float>;
template struct EliminationResult<double>;
