#include<stdio.h>
#include<stdlib.h>

#define ARR_WIDTH 64
#define TILE_WIDTH 4
#define NUM_BLOCK ARR_WIDTH / TILE_WIDTH

__global__ void tile_matrix_multiply(float* A, float* B, float* C, int width)
{
  //printf("kernel\n");
  __shared__ float shareA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shareB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  float temp = 0;

  //Loop over the M and N tiles required to compute the 'P' element
  for(int i = 0; i < width/TILE_WIDTH; i++){
    shareA[ty][tx] = A[row*width + (i*TILE_WIDTH + tx)];
    shareB[ty][tx] = B[(i*TILE_WIDTH + ty)*width + col];
    __syncthreads();
    for(int k = 0; k < TILE_WIDTH; ++k) temp += shareA[ty][k] * shareB[k][tx];
    __syncthreads(); 
  }
  C[row*width + col] = temp;
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
  tile_matrix_multiply<<<dim3(TILE_WIDTH, TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH)>>>(A_g, B_g, C_g, ARR_WIDTH);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  cudaMemcpy(C, C_g, sizeof(float) * ARR_WIDTH*ARR_WIDTH, cudaMemcpyDeviceToHost);
  // printf("\nmatmul C print:\n");
  return 0;
}