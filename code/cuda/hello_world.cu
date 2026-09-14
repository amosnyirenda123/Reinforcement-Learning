#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] From device\n",
           threadIdx.x, blockIdx.x);
}

int main(void) {
    printf("Hello world from host!\n");

    print_from_gpu<<<1,2>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}