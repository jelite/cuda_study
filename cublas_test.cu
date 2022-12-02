#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include<cublas.h>
#include <iostream>

void checkCU(cublasStatus_t status)
{
    if(status != CUBLAS_STATUS_SUCCESS) std::cout << "CUDA ERR" << status << std::endl;
    else std::cout << "CUDA SUCCESS" << status << std::endl;
}

int main()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    const int height = 2;
    const int width = 5;
    float *arr = (float*)malloc(sizeof(float)*height*width);
    float *arr_d;
    float alpha = 2;
    if (!arr) {
        std::cout << "host memory allocation failed" << std::endl;
        return EXIT_FAILURE;
    }
    for(int i = 0; i < height*width; i++) arr[i] = i;

    //cublasAlloc(); deprecated; changed to cudaMalloc()
    cudaStat = cudaMalloc((void**)&arr_d, height*width*sizeof(float));
    if (cudaStat)
    {
        std::cout << "device memory allocation failed" << std::endl;
    }

    stat = cublasCreate_v2(&handle);
    if(stat)
    {
        std::cout << "CUBLAS initializtion failed" << std::endl;
    }


    std::cout << cudaStat << std::endl;
    // checkCU(cublasSetMatrix(width, height, sizeof(float), arr, width, arr_d, width)); ////
    // cublasSscal(width*height, alpha, arr_d, 1); ////
    // checkCU(cublasGetMatrix(width, height, sizeof(float), arr_d, width, arr, width)); ////

    // //cudaThreadSynchronize(); deprecated; changed to cudaDeviceSynchronize()
    // cudaDeviceSynchronize();
    // //cublasFree(); deprecated; changed to cudaFree()
    // cudaFree(arr_d);

    // free(arr);

    return 0;
}