#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void index_check_test() {
  // only print for first thread in each thread block  
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("%d %d\n", blockIdx.x, blockIdx.y);
    printf("%d %d %d %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
  }
}

int main(int argc,char **argv)
{
  size_t total_element = 1024;
  // 16 * 16 thread block
  dim3 block_dim(16, 16);
  // 2 * 2 grid
  size_t length = static_cast<size_t>(sqrt(total_element / (block_dim.x * block_dim.y)));
  dim3 grid_dim(length, length);
  index_check_test<<<grid_dim, block_dim>>>();
  cudaDeviceReset();
  return 0;
}
