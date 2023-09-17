#include <stdio.h>
#include <cuda_runtime.h>

float random_float() {
    return static_cast<float>(abs(random()) % 256);
}

// 没想到被size_t坑了, unsigned用的时候注意一下
__global__ void blur(float* image, float* blur_image, const int width, const int height, const int blur_size) {
    int x_index = blockDim.x * blockIdx.x + threadIdx.x;
    int y_index = blockDim.y * blockIdx.y + threadIdx.y;
    if (x_index < width && y_index < height) {
        float sum_value = 0;
        int count = 0;
        for (int cx = x_index - blur_size;cx <= x_index + blur_size;cx ++) {        
            for (int cy = y_index - blur_size;cy <= y_index + blur_size;cy ++) {
                if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                    int idx = cx + cy * width;
                    sum_value += image[idx];
                    count ++;
                }
            }
        }
        int idx = x_index + y_index * width;
        blur_image[idx] = sum_value / count;
    }
}

int main() {
    // global setup
    const int width = 4 , height = 4, default_device = 0;
    const int total_pixels = width * height;
    const int total_pixel_bytes = total_pixels * sizeof(float);
    cudaSetDevice(default_device);

    // cpu
    float* image_h;
    float* blur_image_d_to_h;
    image_h = (float*) malloc(total_pixel_bytes);
    blur_image_d_to_h = (float*) malloc(total_pixel_bytes);

    // gpu
    float* image_d;
    float* blur_image_d;
    cudaMalloc((void**) &image_d, total_pixel_bytes);
    cudaMalloc((void**) &blur_image_d, total_pixel_bytes);

    // prepare data
    for (int i = 0;i < total_pixels;i ++) {
        image_h[i] = random_float();
    }
    cudaMemcpy(image_d, image_h, total_pixel_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // execute kernel
    dim3 block_dim(2, 2);
    int grid_size = static_cast<int>(sqrt(total_pixels / (block_dim.x * block_dim.y)));
    dim3 grid_dim(grid_size, grid_size);

    const int BLUR_SIZE = 2;
    blur<<<grid_dim, block_dim>>>(image_d, blur_image_d, width, height, BLUR_SIZE);
    cudaMemcpy(blur_image_d_to_h, blur_image_d, total_pixel_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0;i < total_pixels;i ++) {
        printf("%f ", image_h[i]);
    }
    printf("\n");
    for (int i = 0;i < total_pixels;i ++) {
        printf("%f ", blur_image_d_to_h[i]);
    }
    printf("\n");

    // free data
    free(image_h);
    free(blur_image_d_to_h);
    cudaFree(image_d);
    cudaFree(blur_image_d);

    return 0;
}
