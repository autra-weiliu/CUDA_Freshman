#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printThreadIndex(float *A,const int nx,const int ny)
{
  int ix=threadIdx.x+blockIdx.x*blockDim.x;
  int iy=threadIdx.y+blockIdx.y*blockDim.y;
  unsigned int idx=iy*nx+ix;
  printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
          "global index %d ival %f\n",threadIdx.x,threadIdx.y,
          blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}

int main(int argc,char** argv)
{
  cudaSetDevice(0);

  int nx=8,ny=6;
  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);

  //Malloc
  float* A_host=(float*)malloc(nBytes);
  
  for (size_t i = 0;i < nxy;i ++) {
    A_host[i] = (rand() % 1000) * 1.0 / 7;
  }

  //cudaMalloc
  float *A_dev=NULL;
  cudaMalloc((float**) &A_dev,nBytes);

  cudaMemcpy(A_dev,A_host,nBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

  dim3 block(4,2);
  dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);

  printThreadIndex<<<grid,block>>>(A_dev,nx,ny);

  cudaDeviceSynchronize();
  cudaFree(A_dev);
  free(A_host);

  cudaDeviceReset();
  return 0;
}
