#include<stdio.h>
#include<stdlib.h>

#define ARR_WIDTH 64
#define TILE_WIDTH 16
#define NUM_BLOCK ARR_WIDTH / TILE_WIDTH

__global__ void matrix_multiply(float* A, float* B, float* C)
{
    int elem;
    for(int i = 0; i < ARR_WIDTH; i++)
    {
        for(int j = 0; j < ARR_WIDTH; j++)
        {
            for(int k = 0; k < ARR_WIDTH; k++)
            {
                elem += A[i*ARR_WIDTH + k]*B[k*ARR_WIDTH + j];
            }
            C[i*ARR_WIDTH + j] = elem;
        }
    }
}

void random_elem(float* arr, int arr_size)
{
  for(int i = 0; i < arr_size; i++)
  {
    for(int j = 0; j < arr_size; j++)
    {
      arr[i*arr_size + j] = (float)rand()/(float(RAND_MAX/10));
    }
  }
}
void print_array(float* arr, int arr_size)
{
  for(int i = 0; i < arr_size; i++)
  {
    for(int j = 0; j < arr_size; j++)
    {
      printf("%f ", arr[i*arr_size + j]);
    }
    printf("\n");
  }
}

int main(void)
{
  float A[ARR_WIDTH*ARR_WIDTH], B[ARR_WIDTH*ARR_WIDTH], C[ARR_WIDTH*ARR_WIDTH];
  float *A_g, *B_g, *C_g;
  random_elem(A, ARR_WIDTH);
  random_elem(B, ARR_WIDTH);

  cudaMalloc((float**)&A_g, sizeof(float) * ARR_WIDTH*ARR_WIDTH);
  cudaMalloc((float**)&B_g, sizeof(float) * ARR_WIDTH*ARR_WIDTH);
  cudaMalloc((float**)&C_g, sizeof(float) * ARR_WIDTH*ARR_WIDTH);

  cudaMemcpy(A_g, A, sizeof(float) * ARR_WIDTH*ARR_WIDTH, cudaMemcpyHostToDevice);
  cudaMemcpy(B_g, B, sizeof(float) * ARR_WIDTH*ARR_WIDTH, cudaMemcpyHostToDevice);
  cudaMemcpy(C_g, C, sizeof(float) * ARR_WIDTH*ARR_WIDTH, cudaMemcpyHostToDevice);


  // initialize x and y arrays on the kernel
  matrix_multiply<<<1,1>>>(A_g, B_g, C_g);
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  cudaMemcpy(C, C_g, sizeof(float) * ARR_WIDTH*ARR_WIDTH, cudaMemcpyDeviceToHost);
  // printf("\nmatmul C print:\n");
  return 0;
}