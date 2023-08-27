#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char ** argv)
{
  int n_elements = 1024;

  dim3 block(1024);
  dim3 grid(((n_elements - 1) / block.x + 1));
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x = 512;
  grid.x = (n_elements - 1) / block.x + 1;
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x = 256;
  grid.x = (n_elements - 1) / block.x + 1;
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  block.x = 128;
  grid.x = (n_elements - 1) / block.x + 1;
  printf("grid.x %d block.x %d\n",grid.x,block.x);

  cudaDeviceReset();
  return 0;
}
