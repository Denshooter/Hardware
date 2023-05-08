#pragma GCC optimize("O0")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target("avx")

#include <x86intrin.h>
#include <stdio.h>

float scalar_product_seq(__m256 a, __m256 b) {
    float res = 0;
    for (int i = 0; i < 8; i++) {
        res += a[i] * b[i];
    }
    return res;
}

int main(int argc, const char *argv[])
{
    __m256 a = {10,20,30,40,50,60,70,80};
    __m256 b = {1,2,3,4,5,6,7,8};
    __m256 c = _mm256_mul_ps(a, b);
    for (int i = 0; i < 8; i++) {
        printf("%f\n", c[i]);
    }
    return 0;
}
