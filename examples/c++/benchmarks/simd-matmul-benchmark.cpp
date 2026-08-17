#include "ml_lib/math/matrix.h"
#include "config.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

Matrix<double> randomMatrix(int rows, int cols, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double> m(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m(i, j) = dist(rng);
        }
    }
    return m;
}

template <typename Op>
double timeBest(int repeats, Op op) {
    op();
    double best_ms = -1.0;
    for (int r = 0; r < repeats; r++) {
        auto start = std::chrono::steady_clock::now();
        op();
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (best_ms < 0.0 || ms < best_ms) {
            best_ms = ms;
        }
    }
    return best_ms;
}

double benchmarkMatmul(int n, int repeats, std::mt19937& rng) {
    Matrix<double> a = randomMatrix(n, n, rng);
    Matrix<double> b = randomMatrix(n, n, rng);
    return timeBest(repeats, [&]() { Matrix<double> c = a * b; (void)c; });
}

double benchmarkHadamard(int n, int repeats, std::mt19937& rng) {
    Matrix<double> a = randomMatrix(n, n, rng);
    Matrix<double> b = randomMatrix(n, n, rng);
    return timeBest(repeats, [&]() { Matrix<double> c = a.hadamard(b); (void)c; });
}

double benchmarkAdd(int n, int repeats, std::mt19937& rng) {
    Matrix<double> a = randomMatrix(n, n, rng);
    Matrix<double> b = randomMatrix(n, n, rng);
    return timeBest(repeats, [&]() { Matrix<double> c = a + b; (void)c; });
}

}  // namespace

int main() {
    std::printf("ML_USE_SIMD=%d ML_HAS_AVX2=%d", ML_USE_SIMD, ML_HAS_AVX2);
#ifdef _OPENMP
    std::printf(" OMP_THREADS=%d", omp_get_max_threads());
#else
    std::printf(" OMP_THREADS=1(disabled)");
#endif
    std::printf("\n");

    const std::vector<int> sizes = {64, 128, 256, 512, 768, 1024};
    const int repeats = 5;
    std::mt19937 rng(42);

    std::printf("op,size,ms,gflops\n");
    for (int n : sizes) {
        double ms = benchmarkMatmul(n, repeats, rng);
        double gflops = (2.0 * n * n * n) / (ms / 1000.0) / 1e9;
        std::printf("matmul,%d,%.4f,%.4f\n", n, ms, gflops);
    }
    for (int n : sizes) {
        double ms = benchmarkHadamard(n, repeats, rng);
        double gflops = (static_cast<double>(n) * n) / (ms / 1000.0) / 1e9;
        std::printf("hadamard,%d,%.4f,%.4f\n", n, ms, gflops);
    }
    for (int n : sizes) {
        double ms = benchmarkAdd(n, repeats, rng);
        double gflops = (static_cast<double>(n) * n) / (ms / 1000.0) / 1e9;
        std::printf("add,%d,%.4f,%.4f\n", n, ms, gflops);
    }

    return 0;
}
