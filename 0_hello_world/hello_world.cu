#include <stdio.h>

__global__ void hello_world() {
  printf("hello world from gpu\n");
}

int main() {
  printf("hello world from cpu\n");
  hello_world<<<2, 10>>>();
  cudaDeviceReset();
  return 0;
}
