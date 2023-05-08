#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE (1L << 16)

//use __builtin_assume_aligned to tell the compiler that the arrays are aligned
//use __restrict to tell the compiler that the arrays do not overlap
//use __attribute__((aligned(16))) to tell the compiler to align the arrays to 16 bytes

double __attribute__((aligned(16)))vec1[SIZE];
double __attribute__((aligned(16)))vec2[SIZE];

void test(double * __restrict a, double * __restrict b) {
    for (int i = 0; i < SIZE; i++) {
        a[i] += b[i];
    }
}

int main(int argc, const char *argv[])
{
    for (int i = 0; i < SIZE; i++) {
        __builtin_assume_aligned(vec1, 16);
        __builtin_assume_aligned(vec2, 16);
        vec1[i] = i;
        vec2[i] = i;
    }
    test(vec1, vec2);
    return 0;
}
