#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define N 512


void host_add(int *a, int *b, int *c){
    for (size_t i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

__global__ void device_add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
}




void fill_array(int *data){
    for (size_t i = 0; i < N; i++)
        data[i] = i;
}

void print_output(int *a, int *b, int*c) {
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}


int main(void){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    size_t threads_per_block = 8;
    size_t no_of_blocks = N / threads_per_block;

    a = (int*) malloc(size); fill_array(a);
    b = (int*) malloc(size); fill_array(b);
    c = (int*) malloc(size); fill_array(c);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    
    device_add<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    print_output(a, b, c);

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return EXIT_SUCCESS;
}