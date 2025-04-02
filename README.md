# CUDA-Accelerated Monte Carlo Path Tracing with AI Denoising

## Overview

This project implements a real-time Monte Carlo path tracer using CUDA, optimized with deep-learning-based AI denoising to accelerate global illumination rendering. The goal is to achieve physically-based rendering (PBR) in real-time by leveraging BVH acceleration structures and RTX Tensor cores (if available).

## Features

-   **CUDA-accelerated path tracing:** For photorealistic rendering.
-   **Real-time BVH acceleration structures:** For efficient ray traversal.
-   **AI-based denoiser:** Uses a basic CNN model (for now) to reduce noise in rendered images (see **AI Denoiser** section).
-   **Real-time global illumination and indirect lighting:** Achieves realistic lighting effects.
-   **Supports OBJ, GLTF models (Future):** For complex scenes (implementation in progress).

## Why It's Challenging

-   **Warp divergence:** Light paths scatter randomly, making efficient branching and memory access patterns crucial for performance.
-   **BVH optimization:** Constructing and traversing a bounding volume hierarchy (BVH) in real-time requires careful optimization to minimize overhead.
-   **Denoising with AI:** Training and integrating a deep-learning model for fast noise reduction presents significant challenges in terms of model architecture, training data, and real-time performance. The current implementation uses a very basic CNN as a placeholder. A production-ready denoiser would require a more sophisticated model and extensive training.

## CUDA Optimizations

-   **BVH Construction & Traversal:** Optimized to minimize warp divergence. Current implementation includes basic object sorting before BVH construction. Future improvements will include:
    -   Sorting along a space-filling curve (e.g., Morton code).
    -   Using the Surface Area Heuristic (SAH) for better split decisions.
    -   Binning for faster split plane evaluation.
-   **Ray Coherence Optimization:** Future implementation will include memory-friendly ray reordering to improve locality and cache efficiency.
-   **Denoiser on RTX Tensor Cores (Future):** The AI denoiser will be optimized to utilize RTX Tensor Cores for accelerated inference. This will involve converting the PyTorch model to TensorRT.

## Installation

### Prerequisites

-   **CUDA Toolkit 11+**
-   **NVIDIA RTX 30/40-series GPU (recommended for AI denoising)**
-   **CMake 3.18+**
-   **Python 3.8+ (for AI denoising model)**
-   **PyTorch (for training and running the denoiser)**
-   **TensorRT (for optimized denoiser inference - future)**

### Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nmavuso/cuda_monte_carlo_ray_tracer.git](https://github.com/nmavuso/cuda_monte_carlo_ray_tracer.git)
    cd cuda-path-tracer
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Generate build files with CMake:**
    ```bash
    cmake ..
    ```

4.  **Compile the project:**
    ```bash
    make -j$(nproc)
    ```

5.  **Run (after building):**
    ```bash
    ./path_tracer
    ```

## AI Denoiser

The current implementation includes a very basic CNN-based denoiser written in PyTorch. It serves as a placeholder for a more advanced denoiser.

### Training the Denoiser

1.  **Navigate to the `scripts` directory:**
    ```bash
    cd ../scripts
    ```

2.  **Run the training script:**
    ```bash
    python train_denoiser.py --model_dir ../models
    ```
    This will train the basic denoiser using a dummy dataset and save the model to the `models` directory.
    **Important:** The dummy dataset and training process are for illustrative purposes only. Real-world denoising requires a large, high-quality dataset of noisy and corresponding clean images.

### Using the Denoiser

The `main.cu` file integrates a call to the `denoiser.denoise()` function. However, keep in mind that the current denoiser is a placeholder.

### Future Denoiser Improvements

-   **Advanced Model Architectures:** Replace the basic CNN with a more powerful architecture like U-Net, Transformer, or a diffusion model.
-   **High-Quality Training Data:**  Acquire or generate a large dataset of noisy/clean image pairs for training.
-   **TensorRT Conversion:** Convert the trained PyTorch model to TensorRT for optimized inference on NVIDIA GPUs, leveraging Tensor Cores.

