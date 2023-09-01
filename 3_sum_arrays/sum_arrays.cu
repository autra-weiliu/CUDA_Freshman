#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

/***
 *  Looks like gpu programming is interesting, lol, need to tune performance
 */

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error != cudaError_t::cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));\
      exit(1);\
  }\
}

// a + b -> c
void sum_array(float* a, float* b, float* c, size_t len) {
  for (int i = 0;i < len;i ++) {
    c[i] = a[i] + b[i];
  }  
}

__global__ void sum_array_gpu(float* a, float* b, float* c, size_t len) {
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < len) {
    c[index] = a[index] + b[index];
  }
}

void fill_array(float* array, int len) {
  for (size_t i = 0;i < len;i ++) {
    array[i] = (rand() % 10000) * 1.0 / 7;
  }
}

void print_array(float* array, int len) {
  for (size_t i = 0;i < len;i ++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

int main(int argc,char **argv)
{
  // TODO(尝试搞通nvprof的使用)
  size_t len = 10000000;
  size_t bytes_len = sizeof(float) * len;
  float* a_h = (float*) malloc(bytes_len);
  float* b_h = (float*) malloc(bytes_len);
  float* c_h = (float*) malloc(bytes_len);
  float* c_d_to_h = (float*) malloc(bytes_len);

  // prepare test data
  fill_array(a_h, len);
  fill_array(b_h, len);

  // cpu version
  clock_t cpu_start_time, cpu_end_time;
  cpu_start_time = clock();
  sum_array(a_h, b_h, c_h, len);
  cpu_end_time = clock();
  float cpu_time = (static_cast<float>(cpu_end_time - cpu_start_time) / static_cast<float>(CLOCKS_PER_SEC));
  std::cout << "cpu time is: " << cpu_time * 1000 << "ms" << std::endl;

  // setup cuda
  size_t default_device = 0;
  cudaSetDevice(default_device);

  // gpu version
  float* a_d, *b_d, *c_d;
  CHECK(cudaMalloc((float**) &a_d, bytes_len));
  CHECK(cudaMalloc((float**) &b_d, bytes_len));
  CHECK(cudaMalloc((float**) &c_d, bytes_len));

  // copy a, b to device for computation
  cudaMemcpy(a_d, a_h, bytes_len, cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, bytes_len, cudaMemcpyKind::cudaMemcpyHostToDevice);
  
  // gpu version
  float gpu_time;
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, 0));

  size_t block_dim_value = 64;
  dim3 block_dim(block_dim_value);
  size_t grid_dim_value = (len + block_dim_value - 1) / block_dim_value;
  dim3 grid_dim(grid_dim_value);
  sum_array_gpu<<<grid_dim, block_dim>>>(a_d, b_d, c_d, len);

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  std::cout << "gpu time is: " << gpu_time << "ms" << std::endl;
  
  // copy gpu result to cpu
  // length is bytes' length!
  CHECK(cudaMemcpy(c_d_to_h, c_d, bytes_len, cudaMemcpyKind::cudaMemcpyDeviceToHost));

  // compare result
  for (size_t i = 0;i < len;i ++) {
    float diff = abs(c_d_to_h[i] - c_h[i]);
    if (diff > 1e-3) {
      std::cout << diff << std::endl;
    }
  }

  // sync cuda computation
  CHECK(cudaDeviceSynchronize());

  // free gpu devices
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  // free cpu pointers
  free(a_h);
  free(b_h);
  free(c_h);
  free(c_d_to_h);

  // shutdown cuda
  CHECK(cudaDeviceReset());

  return 0;
}
