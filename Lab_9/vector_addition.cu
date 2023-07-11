#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<assert.h>
#include<chrono>
#include<iostream>

#define N 10000000

//Method for adding two Vectors on the CPU
void vector_add_CPU(float *outCPU, float *a, float *b, int n)
{
    for(int i = 0; i < n; i++)
    {
            outCPU[i] = a[i] + b[i];
    }
}


// The Method for adding two vectors on the GPU TODO!
void vector_add_GPU(float *outGPU, float *a, float *b, int n) {
    
    for(int i = 0; i < n; i++)
    {
    	    outGPU[i] = a[i] + b[i];
    }
}


int main(){
    
    // Declare Vektors on Host and Device
    float *a, *b, *outCPU, *outGPU; 

    // Allocate memory. TODO!
    outCPU 	= (float*)malloc(sizeof(float) * N);
    outGPU 	= (float*)malloc(sizeof(float) * N);
    a   	= (float*)malloc(sizeof(float) * N);
    b   	= (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = (float)drand48(); 
	b[i] = (float)drand48();
    }
     
    // Start calculating on CPU:
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    vector_add_CPU(outCPU, a, b, N);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

    
    /////////////////////////////////////////////////////////////////////////
    // TASKS: 
    // 		1. Transform the vector_add_GPU function to a kernel and call 
    //		   the Kernel.
    //		2. Copy the Memory between Host and Device appropiately
    //		3. Test the implementaion.
    // 		4. Change the  Kernel to be executed by parallel threads
    // 		5. Play around with the number of blocks and thread per block
    /////////////////////////////////////////////////////////////////////////

    // Start calculating on GPU: TODO!
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    vector_add_GPU(outGPU, a, b, N);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	
    for(int i = 0; i < N; i++)
    {
    	assert(outCPU[i] == outGPU[i]);
    }

    std::cout << "Time on CPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << "[ms]" << std::endl;
    std::cout << "Time on GPU= " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]" << std::endl;

    // free allocated memory. TODO!
    free(a);
    free(b);
    free(outCPU);
    free(outGPU);
    
}
