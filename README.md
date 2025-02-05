# CUDA-Accelerated Monte Carlo Path Tracing with AI Denoising

## Overview
This project implements a **real-time Monte Carlo ray tracer** using CUDA, optimized with deep-learning-based **AI denoising** to accelerate global illumination rendering. The goal is to achieve **physically-based rendering (PBR)** in real time by leveraging **BVH acceleration structures** and **RTX Tensor cores**.

## Features
- **CUDA-accelerated path tracing** for photorealistic rendering.
- **Real-time BVH acceleration structures** for efficient ray traversal.
- **AI-based denoiser** (Diffusion models / Transformers).
- **Real-time global illumination and indirect lighting**.
- **Supports OBJ, GLTF models** for complex scenes.

## Why It's Challenging
- **Warp divergence**: Since light paths scatter randomly, ensuring efficient branching and memory access patterns is critical.
- **BVH optimization**: Constructing and traversing a **bounding volume hierarchy (BVH)** in real-time.
- **Denoising with AI**: Training and integrating a deep-learning model (e.g., Transformers or diffusion models) for **fast noise reduction**.

## CUDA Optimizations
- **BVH Construction & Traversal**: Optimized to avoid warp divergence.
- **Ray Coherence Optimization**: Memory-friendly ray reordering to improve locality.
- **Denoiser on RTX Tensor Cores**: Uses AI to reduce noise while preserving fine details.

## Installation
### Prerequisites
- **CUDA 11+**  
- **NVIDIA RTX 30/40-series GPU (for AI denoising)**  
- **CMake**  
- **Python 3.8+ (for AI denoising model)**  
- **PyTorch / TensorRT (for deep learning inference)**  

### Clone & Build
```bash
git clone https://github.com/yourusername/cuda-path-tracer.git
cd cuda-path-tracer
mkdir build && cd build
cmake ..
make -j$(nproc)
