#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

void sum_matrix_cpu(float* matrix_1, float* matrix_2, float* matrix_res, const int n, const int m) {
    for (size_t i = 0;i < n;i ++) {
        for (size_t j = 0;j < m;j ++) {
            size_t index = i * m + j;
            matrix_res[index] = matrix_1[index] + matrix_2[index];
        }
    }
}

__global__ void sum_matrix_gpu(float* matrix_1, float* matrix_2, float* matrix_res, const int n, const int m) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < m) {
        size_t index = x + y * m;
        matrix_res[index] = matrix_1[index] + matrix_2[index];
    }
}

void fill_data(float* matrix, const int n, const int m) {
    for (size_t i = 0;i < n;i ++) {
        for (size_t j = 0;j < m;j ++) {
            size_t index = i * m + j;
            matrix[index] = (rand() * 10000) * 1.0 / 13;
        }
    }
}

int main(int argc,char** argv)
{
    // matrix data
    const int n = 1024 , m = 1024;
    const int bytes_len = sizeof(float) * n * m;

    // cpu data
    float* matrix_1_h;
    float* matrix_2_h;
    float* matrix_res_h;
    float* matrix_res_d_to_h;

    matrix_1_h = (float*) malloc(bytes_len);
    matrix_2_h = (float*) malloc(bytes_len);
    matrix_res_h = (float*) malloc(bytes_len);
    matrix_res_d_to_h = (float*) malloc(bytes_len);

    // gpu data
    float* matrix_1_d;
    float* matrix_2_d;
    float* matrix_res_d;

    cudaMalloc((float**) &matrix_1_d, bytes_len);
    cudaMalloc((float**) &matrix_2_d, bytes_len);
    cudaMalloc((float**) &matrix_res_d, bytes_len);

    size_t default_device = 0;
    cudaSetDevice(default_device);

    // fill data
    fill_data(matrix_1_h, n, m);
    fill_data(matrix_2_h, n, m);

    // cpu computation
    sum_matrix_cpu(matrix_1_h , matrix_2_h, matrix_res_h, n , m);

    // gpu computation
    cudaMemcpy((float**) matrix_1_d, matrix_1_h, bytes_len, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy((float**) matrix_2_d, matrix_2_h, bytes_len, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy((float**) matrix_res_d, matrix_res_h, bytes_len, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 block_dim(64, 64);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (m + block_dim.y - 1) / block_dim.y);

    sum_matrix_gpu<<<grid_dim, block_dim>>>(matrix_1_d, matrix_2_d, matrix_res_d, n , m);
    cudaMemcpy((float**) matrix_res_d_to_h, matrix_res_d, bytes_len, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // compare
    for (size_t i = 0;i < n;i ++) {
        for (size_t j = 0;j < m;j ++) {
            size_t index = i * m + j;
            if (matrix_res_h[index] != matrix_res_d_to_h[index]) {
                std::cerr << "error found" << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "succeeded" << std::endl;

    // free data
    free(matrix_1_h);
    free(matrix_2_h);
    free(matrix_res_h);
    free(matrix_res_d_to_h);

    cudaFree(matrix_1_d);
    cudaFree(matrix_2_d);
    cudaFree(matrix_res_d);

    // shutdown
    cudaDeviceReset();
    return 0;
}
