#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 2048
#define BLOCK_SIZE 32



__global__ void matrix_transpose_shared(int *input, int *output)
{

    __shared__ int tile[BLOCK_SIZE][BLOCK_SIZE];

   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }

    
    __syncthreads();

    int transposedX = blockIdx.y * blockDim.x + threadIdx.x;
    int transposedY = blockIdx.x * blockDim.y + threadIdx.y;


    
    if (transposedX < N && transposedY < N)
    {
        output[transposedY * N + transposedX] =
            tile[threadIdx.x][threadIdx.y];
    }
}


void fill_array(int *data)
{
    for (int i = 0; i < N * N; i++)
    {
        data[i] = i;
    }
}



int verify_transpose(int *input, int *output)
{
    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            int expected = input[y * N + x];
            int actual = output[x * N + y];

            if (actual != expected)
            {
                printf(
                    "Verification failed at (%d, %d): "
                    "expected %d, got %d\n",
                    x, y, expected, actual
                );

                return 0;
            }
        }
    }

    return 1;
}



//
void print_small_matrix(int *matrix)
{
    int size = 8;

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            printf("%6d ", matrix[y * N + x]);
        }

        printf("\n");
    }
}



int main(void)
{
    int *a;
    int *b;

    int *d_a;
    int *d_b;

    int size = N * N * sizeof(int);


   

    a = (int *)malloc(size);
    b = (int *)malloc(size);

    if (a == NULL || b == NULL)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }


   

    fill_array(a);


   

    cudaError_t err;

    err = cudaMalloc((void **)&d_a, size);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaMalloc d_a failed: %s\n",
            cudaGetErrorString(err)
        );

        return EXIT_FAILURE;
    }


    err = cudaMalloc((void **)&d_b, size);

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaMalloc d_b failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_a);

        return EXIT_FAILURE;
    }


 

    err = cudaMemcpy(
        d_a,
        a,
        size,
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaMemcpy H2D failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_a);
        cudaFree(d_b);

        return EXIT_FAILURE;
    }


    

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );


    printf("Matrix size : %d x %d\n", N, N);
    printf("Block size  : %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Grid size   : %d x %d\n\n",
           gridSize.x,
           gridSize.y);



    printf("Running shared-memory transpose...\n");

    matrix_transpose_shared<<<gridSize, blockSize>>>(
        d_a,
        d_b
    );


    // Check kernel launch.
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "Shared kernel launch error: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_a);
        cudaFree(d_b);

        free(a);
        free(b);

        return EXIT_FAILURE;
    }


    // Wait for GPU execution to finish.
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "Shared kernel execution error: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_a);
        cudaFree(d_b);

        free(a);
        free(b);

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // Copy shared-memory result back to host
    // --------------------------------------------------------

    err = cudaMemcpy(
        b,
        d_b,
        size,
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess)
    {
        fprintf(
            stderr,
            "Shared D2H copy failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_a);
        cudaFree(d_b);

        free(a);
        free(b);

        return EXIT_FAILURE;
    }


    // --------------------------------------------------------
    // Verify shared-memory result
    // --------------------------------------------------------

    if (verify_transpose(a, b))
    {
        printf("Shared-memory transpose: PASSED\n\n");
    }


    // --------------------------------------------------------
    // Show a small portion of the result
    // --------------------------------------------------------

    printf("First 8 x 8 elements of transposed matrix:\n\n");

    print_small_matrix(b);


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    free(a);
    free(b);

    cudaFree(d_a);
    cudaFree(d_b);

    return EXIT_SUCCESS;
}