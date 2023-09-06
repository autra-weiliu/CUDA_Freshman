#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel_1(float* values, const int len) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len) {
		if (index % 2 == 0) {
			values[index] = 1;
		} else {
			values[index] = 2;
		}
	}
}

__global__ void kernel_2(float* values, const int len) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int wrap_index = threadIdx.x / warpSize;
	if (index < len) {
		if (wrap_index % 2 == 0) {
			values[index] = 1;;
		} else {
			values[index] = 2;
		}
	}
}

__global__ void warmup_kernel(float* values, const int len) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < len) {
		if (index % 2 == 0) {
			values[index] = 3;
		} else {
			values[index] = 4;
		}
	}
}

int main(int argc, char **argv)
{
	size_t default_device = 0;
	cudaSetDevice(default_device);
	
	// gpu data
	float* a_d;
	float* b_d;
	size_t len = 100000000;
	size_t bytes_len = len * sizeof(float);

	// malloc data
	cudaMalloc((float**) &a_d, bytes_len);
	cudaMalloc((float**) &b_d, bytes_len);

	// try different kernel
	dim3 block_dim(512);
	dim3 grid_dim((len + block_dim.x - 1) / block_dim.x);

	// warmup first
	warmup_kernel<<<grid_dim, block_dim>>>(a_d, len);
	cudaDeviceSynchronize();

	// execute kernel1
	float gpu_time;
	cudaEvent_t start_1, stop_1, start_2, stop_2;
  	cudaEventCreate(&start_1);
  	cudaEventCreate(&stop_1);
  	cudaEventRecord(start_1, 0);

	kernel_1<<<grid_dim, block_dim>>>(a_d, len);
	cudaDeviceSynchronize();

	cudaEventRecord(stop_1, 0);
	cudaEventSynchronize(stop_1);
	cudaEventElapsedTime(&gpu_time, start_1, stop_1);
	cudaEventDestroy(start_1);
	cudaEventDestroy(stop_1);

	printf("gpu time1: %fms \n", gpu_time);

	// execute kernel2
	cudaEventCreate(&start_2);
  	cudaEventCreate(&stop_2);
  	cudaEventRecord(start_2, 0);

	kernel_2<<<grid_dim, block_dim>>>(a_d, len);
	cudaDeviceSynchronize();

	cudaEventRecord(stop_2, 0);
	cudaEventSynchronize(stop_2);
	cudaEventElapsedTime(&gpu_time, start_2, stop_2);
	cudaEventDestroy(start_2);
	cudaEventDestroy(stop_2);

	printf("gpu time2: %fms \n", gpu_time);

	return EXIT_SUCCESS;
}
