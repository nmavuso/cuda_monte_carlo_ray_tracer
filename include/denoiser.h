#ifndef DENOISER_H
#define DENOISER_H

#include <cuda_runtime.h>

// Placeholder for the AI denoiser interface.
// In a real implementation, this would involve loading a trained
// PyTorch/TensorRT model and performing inference.

class Denoiser {
public:
    Denoiser(const char* modelPath);
    ~Denoiser();

    // Denoiser inference
    void denoise(float* inputImage, float* outputImage, int width, int height);

private:
    // Placeholder for model-specific data (e.g., TensorRT engine)
    void* modelData;
};

#endif
