#pragma GCC optimize("O3")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper");
#pragma GCC target("avx")

#define _GNU_SOURCE

#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#define run_benchmark 0

// number bytes in the simd vector
#define SIMD_BYTES 32
// number of floats in a simd vector
#define SIMD_FLOATS (SIMD_BYTES / sizeof(float))

// data type for matrix
typedef struct {
    // number of rows, number of columns
    int rows, cols;
    // data: row-by-row
    float *data;
} Matrix;

void matrixAlloc(Matrix *m, int rows, int cols)
{
    m->rows = rows;
    m->cols = cols;
    // posix_memalign: 16 byte / 32 byte alignment
    // m->data = malloc(rows * cols * sizeof(float));
    if (posix_memalign((void**)&(m->data), SIMD_BYTES, rows * cols * sizeof(float))) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(-1);
    }
}

void matrixClear(Matrix *m)
{
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
}

void matrixRandom(Matrix *m)
{
    // random initialization
    for (int i = 0; i < m->cols * m->rows; i++) {
        m->data[i] = drand48();
    }
}

void matrixPrint(Matrix *a)
{
    for (int r = 0; r < a->rows; r++) {
        for (int c = 0; c < a->cols; c++) {
            printf("%6.2f ", a->data[a->cols * r + c]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrixAddSIMD(Matrix *a, Matrix *b, Matrix *c)
{
    if ((a->rows != b->rows) || (a->cols != b->cols)) {
        fprintf(stderr, "matrixAdd: input size mismatch\n");
        exit(-1);
    }


#if 0
    // number of elements
    int sz = a->rows * a->cols;

    // sequential version with alignment considerations (no
    // seq. preamble, but postamble)
    float *a1, *b1, *c1;
    a1 = __builtin_assume_aligned(a->data, SIMD_BYTES);
    b1 = __builtin_assume_aligned(b->data, SIMD_BYTES);
    c1 = __builtin_assume_aligned(c->data, SIMD_BYTES);  
    for (int i = 0; i < sz; i++) {
        c1[i] = a1[i] + b1[i];
    }
#endif 

    // TODO: a) YOUR CODE HERE

}

void matrixMultSeq(Matrix *a, Matrix *b, Matrix *c)
{
    if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols) {
        fprintf(stderr, "matrixAdd: input size mismatch\n");
        exit(-1);
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                c->data[i * c->cols + j] += a->data[i * a->cols + k] * b->data[b->cols * k + j];
            }
        }
    }
}

void matrixMultSIMD(Matrix *a, Matrix *b, Matrix *c)
{
    if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols) {
        fprintf(stderr, "matrixAdd: input size mismatch\n");
        exit(-1);
    }

    // TODO: b) YOUR CODE HERE

}

double timespec_diff(struct timespec ts1, struct timespec ts2) {
    return (double)(ts2.tv_sec - ts1.tv_sec) + (double)((ts2.tv_nsec - ts1.tv_nsec)/1e9f);
}

int main(int argc, char *argv[])
{
    srand48(time(NULL));

    Matrix a, b, c, d;

    int m = 9;

    // intialize matrix a
    printf("Matrix a\n");
    matrixAlloc(&a, m, m);
    matrixRandom(&a);
    matrixPrint(&a);

    // initialize matrix b
    printf("Matrix b\n");
    matrixAlloc(&b, m, m);
    matrixRandom(&b);
    matrixPrint(&b);

    // c = a + b
    printf("Matrix c = a + b\n");
    matrixAlloc(&c, m, m);
    matrixAddSIMD(&a, &b, &c);
    matrixPrint(&c);

    // d = a * b
    printf("Sequential; d = a * b\n");
    matrixAlloc(&d, m, m);
    matrixMultSeq(&a, &b, &d);
    matrixPrint(&d);

    // d = a * b
    printf("SIMD; d = a * b\n");
    matrixClear(&d);
    matrixMultSIMD(&a, &b, &d);
    matrixPrint(&d);

#if (run_benchmark == 1)

    // benchmark
    FILE *benchmark = fopen("benchmark.csv", "w");
    Matrix e, f, g, h;
    int n = 64;

    printf("Init Benchmark.\n");

    for (int n = 3; n <= 64; n++) {
        matrixAlloc(&e, n, n);
        matrixRandom(&e);

        matrixAlloc(&f, n, n);
        matrixRandom(&f);

        matrixAlloc(&g, n, n);
        matrixClear(&g);

        matrixAlloc(&h, n, n);
        matrixClear(&h);
        
        printf("Matrices allocated.\n");

        struct timespec ts0, ts1;

        double seq, simd;

        printf("Starting with benchmark n = %d\n", n);

        clock_gettime(CLOCK_REALTIME, &ts0);
        for (int i = 0; i < 10000; i++) {
            matrixMultSeq(&e, &f, &g);
        }
        clock_gettime(CLOCK_REALTIME, &ts1);
        seq = timespec_diff(ts0, ts1);

        printf("Done sequential.\n");

        clock_gettime(CLOCK_REALTIME, &ts0);
        for (int i = 0; i < 10000; i++) {
            matrixMultSIMD(&e, &f, &h);
        }
        clock_gettime(CLOCK_REALTIME, &ts1);
        simd = timespec_diff(ts0, ts1);

        printf("Done SIMD.\n");

        fprintf(benchmark, "%i, %f, %f\n", n, seq, simd);

        free(e.data);
        free(f.data);
        free(g.data);
        free(h.data);
    }

    fclose(benchmark);

#endif
    return 0;
}
