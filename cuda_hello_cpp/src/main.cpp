// Plain C++ host program. Compiled with g++ (no CUDA headers here).
//
// All GPU work happens inside launch_vector_add(), which is implemented in
// add_kernel.cu and compiled by nvcc. The two object files are linked
// together against the CUDA runtime by the Makefile.

#include "add_kernel.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    constexpr int n = 1 << 20;  // 1,048,576 elements

    std::vector<float> a(n), b(n), c(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    launch_vector_add(a.data(), b.data(), c.data(), n);

    // Spot-check a few elements and validate the full result.
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::fprintf(stderr,
                             "mismatch at %d: got %f, expected %f\n",
                             i, c[i], expected);
            }
            ++errors;
        }
    }

    std::printf("vector_add: n=%d, c[0]=%.1f, c[1]=%.1f, c[%d]=%.1f\n",
                n, c[0], c[1], n - 1, c[n - 1]);

    if (errors != 0) {
        std::fprintf(stderr, "FAILED: %d mismatches\n", errors);
        return EXIT_FAILURE;
    }
    std::printf("OK\n");
    return EXIT_SUCCESS;
}
