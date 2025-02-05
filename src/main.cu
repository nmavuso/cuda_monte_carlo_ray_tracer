#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "camera.h"
#include "sphere.h"
#include "hitable_list.h"
#include "material.h"
#include "bvh.h"
#include "utils.h"
#include "denoiser.h"


// CUDA kernel for rendering each pixel
__global__ void render_kernel(vec3* fb, int width, int height, int ns, camera cam, hitable* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= width) || (j >= height)) return;

    int image_index = j * width + i;
    curandState local_state = state;

    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_state)) / float(width);
        float v = float(j + curand_uniform(&local_state)) / float(height);
        ray r = cam.get_ray(u, v);
        col += color(r, world, 0, &local_state);
    }
    state = local_state;
    col /= float(ns);
    col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
    fb[image_index] = col;
}

// CUDA kernel for initializing the curand state
__global__ void init_curand_kernel(int seed, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= width) || (j >= height)) return;
    int index = j * width + i;
    curand_init(seed, index, 0, &state);
}

// Function to compute the color of a ray
__device__ vec3 color(const ray& r, hitable* world, int depth, curandState* local_state) {
    hit_record rec;
    if (world->hit(r, 0.001, FLT_MAX, rec)) {
        ray scattered;
        vec3 attenuation;
        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return attenuation * color(scattered, world, depth + 1, local_state);
        }
        else {
            return vec3(0, 0, 0);
        }
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

int main() {
    int nx = 800;
    int ny = 400;
    int ns = 10; // Number of samples per pixel

    // Allocate Unified Memory
    vec3* cpu_fb;
    vec3* gpu_fb;
    cudaMallocManaged(&cpu_fb, nx * ny * sizeof(vec3));
    cudaMallocManaged(&gpu_fb, nx * ny * sizeof(vec3));

    // Initialize curand
    init_curand_kernel << <((nx * ny + 255) / 256), 256 >> > (1234, nx, ny);

    // Create scene objects
    hitable* list[5];
    list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.8, 0.3, 0.3)));
    list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
    list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.3));
    list[3] = new sphere(vec3(-1, 0, -1), 0.5, new metal(vec3(0.8, 0.8, 0.8), 1.0));
    list[4] = new sphere(vec3(0, 1, -1), 0.5, new lambertian(vec3(0.1, 0.3, 0.8)));
    hitable* world = new hitable_list(list, 5);

    // Create camera
    vec3 lookfrom(3, 3, 2);
    vec3 lookat(0, 0, -1);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 2.0;
    camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny));

    // Render
    dim3 blocks(nx / 16, ny / 16);
    dim3 threads(16, 16);
    render_kernel << <blocks, threads >> > (gpu_fb, nx, ny, ns, cam, world);

    // Ensure all computations are finished
    cudaDeviceSynchronize();

    // Copy results from GPU to CPU
    for (int i = 0; i < nx * ny; i++) {
        cpu_fb[i] = gpu_fb[i];
    }

    // Denoise the image (using a placeholder for the actual AI denoiser)
    // In a real scenario, you would call the denoiser here.
    Denoiser denoiser("../models/denoiser_model");  // Replace with the actual path
    float* denoised_image; // This would be managed by the denoiser
    cudaMallocManaged(&denoised_image, nx * ny * 3 * sizeof(float)); //float3 image

    // Convert vec3 to float array for denoising
    float* float_image;
    cudaMallocManaged(&float_image, nx * ny * 3 * sizeof(float));
    for(int i=0; i < nx * ny; ++i){
        float_image[i*3] = cpu_fb[i].x();
        float_image[i*3+1] = cpu_fb[i].y();
        float_image[i*3+2] = cpu_fb[i].z();
    }

    denoiser.denoise(float_image, denoised_image, nx, ny);

    // Convert back to vec3 for saving the image.
    for(int i=0; i < nx * ny; i++){
        cpu_fb[i] = vec3(denoised_image[i*3], denoised_image[i*3+1], denoised_image[i*3+2]);
    }

    // Save the image to a PPM file
    std::ofstream image_file("output.ppm");
    image_file << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            int ir = int(255.99 * cpu_fb[j * nx + i].r());
            int ig = int(255.99 * cpu_fb[j * nx + i].g());
            int ib = int(255.99 * cpu_fb[j * nx + i].b());
            image_file << ir << " " << ig << " " << ib << "\n";
        }
    }
    image_file.close();

    // Free Unified Memory
    cudaFree(cpu_fb);
    cudaFree(gpu_fb);
    cudaFree(float_image);
    cudaFree(denoised_image);

    // Clean up scene objects
    for (int i = 0; i < 5; i++) {
        delete list[i];
    }
    delete world;

    return 0;
}
