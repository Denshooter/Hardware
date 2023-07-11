#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<chrono>
#include<iostream>

// Define the number of elements in the vectors
#define N 100000000

/**
 * Function for adding two Vectors on the CPU
 * Each element of the vectors is added sequentially
 * The result is written to the output vector
*/
void vector_add_CPU(float *outCPU, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
            outCPU[i] = a[i] + b[i];
    }
}


/**
 * Kernel for adding two Vectors on the GPU
 * Each thread gets a unique ID (tid)
 * Each thread adds two elements of the vectors
 * Each thread writes the result to the output vector
 * The number of threads is equal to the number of elements in the vectors
*/
__global__ void vector_add_GPU(float *outGPU, float *a, float *b, int n) {
	
    // gridDim:   represents the dimensions of the grid
    // blockIdx:  represents index of current block within the grid in upto 3 dimensions
    // blockDim:  represents dimension of each block. Provides number of threads for the current block in upto 3 dimensions
    // threadIdx: represents current index of the current thread within its block in upto 3 dimensions  
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) 
    {
    	outGPU[tid] = a[tid] + b[tid];
    }
}

/**
 * 1. Allocate memory on the Host and Device
 * 2. Initialize the Vectors on the Host
 * 3. Copy the Vectors from the Host to the Device
 * 4. Call the Kernel
 * 5. Copy the Result from the Device to the Host
 * 6. Free the allocated memory on the Host and Device
 * 7. Measure the Time 
 * 8. Verify the Result
*/
int main(){
    
    // Declare Vektors on Host and Device
    float *a, *b, *outCPU, *outGPU; 

    outCPU 	= (float*)malloc(sizeof(float) * N);

    // cudaMallocManaged expects void** type argument to store 
    // the allocated memeory address. (void**)&ouGPU  passes the address of 
    // the outGPU pointer to the method.
    cudaMallocManaged((void**)&a, sizeof(float) * N); 
    cudaMallocManaged((void**)&b, sizeof(float) * N);
    cudaMallocManaged((void**)&outGPU, sizeof(float) * N);

    // Initialize the Vectors on the Host
    for(int i = 0; i < N; i++){
        a[i] = (float)drand48(); 
    	b[i] = (float)drand48(); 
    }
     
    // Start calculating on CPU
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    vector_add_CPU(outCPU, a, b, N);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

    // Start calculating on GPU 
    int blockSize = 32; // 32 threads per block
    int gridSize = ((N + blockSize - 1) / blockSize); // To ensure there are enough blocks for the input size
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    vector_add_GPU<<<gridSize,blockSize>>>(outGPU, a, b, N); // <<<gridSize,blockSize>>> is the syntax for calling a kernel
    cudaDeviceSynchronize(); // Wait for the GPU to finish before accessing on host
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	
    // Verify the Result
    for(int i = 0; i < N; i++)
    {
    	assert(outCPU[i] == outGPU[i]);
    }

    std::cout << "Time on CPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << "[ms]" << std::endl;
    std::cout << "Time on GPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]" << std::endl;

    // free allocated memory
    cudaFree(a); // cudaFree is used to free memory allocated with cudaMallocManaged
    cudaFree(b);
    free(outCPU);
    cudaFree(outGPU);
    
}
