#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main(){
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量" << devProp.multiProcessorCount << std::endl;
}
