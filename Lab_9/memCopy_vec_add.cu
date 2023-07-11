#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>

#define N 10000000

// Visualization for vectors with few entries
void print_result(float *out, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
	    printf("%f + %f = %f\n", a[i], b[i], out[i]);
    }
	
}

void vector_add_CPU(float *outCPU, float *a, float *b, int n)
{
	for(int i = 0; i < n; i++)
	{
		outCPU[i] = a[i] + b[i];
	}
}


// The Method for adding two vectors
__global__ void vector_add_GPU(float *outGPU, float *a, float *b, int n) {
    

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
    {
    	outGPU[thread_id] = a[thread_id] + b[thread_id];
    }
    /** int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index = 0; i < n; i = i + stride){
        out[i] = a[i] + b[i];
    }
    */
}


int main(){
    
    // Declare Vektors on Host
    float *a, *b, *outCPU, *outGPU; 
    //Declare Vectors on device
    float *device_a, *device_b, *device_out;

    // Allocate memory
    a   	= (float*)malloc(sizeof(float) * N);
    b   	= (float*)malloc(sizeof(float) * N);
    outCPU 	= (float*)malloc(sizeof(float) * N);
    outGPU 	= (float*)malloc(sizeof(float) * N);
	
    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = (float)drand48(); b[i] = (float)drand48();
    }
    
    vector_add_CPU(outCPU, a, b, N);

    // Allocate device memory
    cudaMalloc((void**)&device_a, sizeof(float) * N);
    cudaMalloc((void**)&device_b, sizeof(float) * N);
    cudaMalloc((void**)&device_out, sizeof(float) * N);

    // Transfer from Host to Device Memory
    cudaMemcpy(device_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, outGPU, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // Call the Kernel function
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add_GPU<<<grid_size, block_size>>>(device_out, device_a, device_b, N);

    // Transfer result from Device to Host Memory
    cudaMemcpy(outGPU, device_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    //print_result(out, a, b, N); 


    for(int i = 0; i< N; i++)
    {
    	assert(outGPU[i] == outCPU[i]);
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_out);
    free(a);
    free(b);
    free(outGPU);
    free(outCPU);
    
}
