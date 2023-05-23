#include <stdio.h>
#include <x86intrin.h>

// Function to print the values of a __m256 vector
void print_vector(__m256 vec)
{
    for (int i = 0; i < 8; i++) {
        //6.2f means 6 characters wide, 2 decimal places
        printf("%6.2f ", vec[i]);
    }
    printf("\n");
}

__m256 compute_ci(__m256 a, __m256 b)
{
    __m256 mask = _mm256_cmp_ps(a, b, _CMP_GE_OS);  // Compare a >= b
    return _mm256_blendv_ps(a, a-b, mask);            // Blend based on the mask
}

__m256 compute_ci2(__m256 a)
{
    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(a, zero, _CMP_GE_OQ);  // Compare a >= 0
    __m256 sqrt_a = _mm256_sqrt_ps(a);                // Square root of a
    return _mm256_blendv_ps(zero, sqrt_a, mask);      // Blend based on the mask
}

int main()
{
    // Test case for compute_ci with two vectors
    __m256 a = _mm256_set_ps(2.0f, 5.0f, 3.0f, 8.0f, 6.0f, 4.0f, 7.0f, 1.0f);
    __m256 b = _mm256_set_ps(7.0f, 3.0f, 5.0f, 2.0f, 1.0f, 8.0f, 4.0f, 6.0f);
    __m256 ci = compute_ci(a, b);

    printf("a: ");
    print_vector(a);
    printf("b: ");
    print_vector(b);

    printf("c: ");
    print_vector(ci);


    printf("\n");
    // Test case for compute_ci2 with a single vector
    __m256 x = _mm256_set_ps(-1.0f, 4.0f, -2.0f, 9.0f, 0.0f, -5.0f, 3.0f, -6.0f);
    __m256 ci2 = compute_ci2(x);

    printf("x: ");
    print_vector(x);

    printf("c: ");
    print_vector(ci2);

    return 0;
}
