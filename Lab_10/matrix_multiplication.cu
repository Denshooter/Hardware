#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<chrono>
#include<iostream>

// Define the number of elements in the matrices
#define N 1000
#define TILE_WIDTH 32

// Function for multiplying two Matrices on the CPU
void matrix_mul_CPU(float *outCPU, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            float sum = 0;
            for(int k = 0; k < n; k++)
            {
                sum += a[i * n + k] * b[k * n + j];
            }
            outCPU[i * n + j] = sum;
        }
    }
}

// Kernel for multiplying two Matrices on the GPU
__global__ void matrix_mul_GPU(float *outGPU, float *a, float *b, int n) {
    __shared__ float A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0;

    for (int ph = 0; ph < n/TILE_WIDTH; ++ph) {
        if (row < n && ph*TILE_WIDTH+tx < n)
            A[ty][tx] = a[row*n + ph*TILE_WIDTH + tx];
        else
            A[ty][tx] = 0.0;

        if (col < n && ph*TILE_WIDTH+ty < n)
            B[ty][tx] = b[(ph*TILE_WIDTH + ty)*n + col];
        else
            B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += A[ty][k] * B[k][tx];

        __syncthreads();
    }

    if(row < n && col < n)
        outGPU[row*n + col] = sum;
}

// Function to check for CUDA errors
void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }                         
}

int main(){
    // Declare Matrices on Host and Device
    float *a, *b, *outCPU, *outGPU; 

    outCPU 	= (float*)malloc(sizeof(float) * N * N);

    // cudaMallocManaged expects void** type argument to store 
    // the allocated memeory address. (void**)&ouGPU  passes the address of 
    // the outGPU pointer to the method.
    cudaMallocManaged((void**)&a, sizeof(float) * N * N); 
    cudaMallocManaged((void**)&b, sizeof(float) * N * N);
    cudaMallocManaged((void**)&outGPU, sizeof(float) * N * N);

    // Initialize the Matrices on the Host
    for(int i = 0; i < N * N; i++){
        a[i] = (float)drand48(); 
    	b[i] = (float)drand48(); 
    }
     
    // Start calculating on CPU
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    matrix_mul_CPU(outCPU, a, b, N);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

    // Start calculating on GPU 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    matrix_mul_GPU<<<dimGrid,dimBlock>>>(outGPU, a, b, N);
    checkCUDAError("kernel invocation");
    cudaDeviceSynchronize(); // Wait for the GPU to finish before accessing on host
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

    // Verify the Result
    for(int i = 0; i < N * N; i++)
    {
        if (outCPU[i] != outGPU[i])
        {
            std::cout << "Error at index " << i << std::endl;
            std::cout << "CPU: " << outCPU[i] << std::endl;
            std::cout << "GPU: " << outGPU[i] << std::endl;
        }
    }

    std::cout << "Time on CPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << "[ms]" << std::endl;
    std::cout << "Time on GPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]" << std::endl;

    // free allocated memory
    cudaFree(a); // cudaFree is used to free memory allocated with cudaMallocManaged
    cudaFree(b);
    free(outCPU);
    cudaFree(outGPU);
    
}
