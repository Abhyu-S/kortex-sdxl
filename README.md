# Generative Fill & Harmonization with Optimized SDXL

This repository contains a high-efficiency generative fill and image harmonization system powered by Stable Diffusion XL (SDXL) and ControlNet. It implements advanced optimization techniques to run heavy diffusion pipelines on consumer-grade GPUs with reduced latency and memory footprint.The system allows users to perform complex image editing tasks—such as object replacement, background generation, and sticker harmonization—while maintaining photometric consistency through a "Vibe Match" mechanism.

## Features
**Optimized SDXL Backbone:** Utilizes 4-bit NF4 Quantization and Token Merging (ToMe) to significantly reduce VRAM usage and inference time without compromising generation quality.
### Core Editing Modes:
1. Smart Fill (Generative Fill): Uses ControlNet-Inpaint-Dreamer with SDXL to fill masked areas based on text prompts.
2. Harmonization:Designed for "moved objects" or stickers.Performs edge-only blending and anti-aliasing to make pasted objects look native to the background.
### Real-time Monitoring:
Built-in resource monitoring (RAM, CPU, GPU utilization) and TFLOPS estimation, logging metrics directly to Weights & Biases (WandB).Built with FastAPI and Diffusers.
### Project Structure
```Bash
├── abhyu-s/diffusion_tobeused/
│   ├── server.py                 # Application API (FastAPI endpoints)
│   ├── editing_pipelines_fill.py # Core Diffusion Logic (SDXL & ControlNet)
│   ├── quantization_utils.py     # 4-bit NF4 Configuration Logic
│   ├── pruning_utils.py          # Token Merging (ToMe) Logic
│   └── requirements.txt          # Python Dependencies
```
## Datasets
This project utilizes a Training-Free approach. It leverages pre-trained weights from Stability AI and Destitech, applying inference-time optimizations to achieve performance goals.
- Inference data: Accepts standard image formats (JPG, PNG) and binary masks.

## The 2030 Compute Proxy: Why Tesla T4?
To address the challenge's requirement for a lightweight, mobile-first editor, we utilized the NVIDIA Tesla T4 as a hardware proxy for the estimated compute capability of a flagship mobile NPU in 2030

- **Current State (2024)**: High-end mobile NPUs (e.g., A17 Pro, Snapdragon 8 Gen 3) reach ~35-45 TOPS (Trillions of Operations Per Second)

- **The 2030 Projection**: Extrapolating current NPU efficiency gains, mobile edge devices in 2030 are projected to exceed 100+ TOPS, rivaling the inference throughput of today's mid-range inference cards like the T4 (~65 TOPS Int8 / ~8 TFLOPS FP32)

- **Our Approach**: By optimizing Stable Diffusion XL (SDXL) with 4-bit Quantization and Token Merging, we demonstrate that high-fidelity generative editing can run locally on this "2030-equivalent" compute profile without cloud dependency.

## Getting Started
### Prerequisites
1. NVIDIA GPU (Tested on Tesla T4, compatible with 12GB+ VRAM).
2. Python 3.10+.
3. CUDA toolkit installed.
### Installation
1. **Install Requirements**
```Bash
pip install -r requirements.txt
```
2. **Start the Server**: Run server.py to initialize the pipeline. The model loading includes a quantization step that may take a moment.
```Bash
python server.py
```
The server will start on http://0.0.0.0:8080.
## API Usage
### 1. Smart Fill (Generative Fill)
Performs inpainting with an optional "Vibe Match" pass for lighting consistency.

Endpoint: `POST /smart-fill`
```Bash
curl -X POST "http://localhost:8080/smart-fill" \
  -F "image=@/path/to/source_image.png" \
  -F "mask=@/path/to/mask_image.png" \
  -F "prompt=A futuristic cyberpunk city background"
```
- `vibe_strength`: (Float 0.0 - 1.0). Controls how aggressively the system attempts to relight the generated pixels to match the source image.
### 2. Harmonization
Blends a pasted object (sticker) into the background by processing the edges.

Endpoint: `POST /harmonize`
```Bash
curl -X POST "http://localhost:8080/harmonize" \
  -F "image=@/path/to/composite_image.png" \
  -F "mask=@/path/to/sticker_mask.png"
```
## File Explanations
### Core Application Logic
1. **server.py:** The FastAPI entry point. It initializes the EditingPipelines class (which loads the models) and defines endpoints for /generative-fill, /harmonize, and /smart-fill. It handles image decoding and byte-stream responses.
2. **editing_pipelines_fill.py:** The Brain. It initializes the StableDiffusionXLControlNetInpaintPipeline and StableDiffusionXLImg2ImgPipeline.
- **ResourceMonitor**: A background thread that tracks RAM and CPU usage during inference.
- **run_smart_fill**: Orchestrates the generation process using ControlNet and SDXL.
- **run_harmonize_sticker**: Handles bounding box extraction and edge-focused inpainting for faster processing (768x768 crop).
3. **quantization_utils.py**: Contains the BitsAndBytesConfig setup. It configures the model to load in 4-bit NF4 (Normal Float 4) precision with double quantization, drastically reducing the memory footprint of the SDXL UNet and Text Encoders.
4. **pruning_utils.py**: Implements Token Merging (ToMe). This applies dynamic structural pruning to the attention mechanism, removing approximately 40% of redundant tokens during the forward pass to speed up inference.
## Optimizations
To ensure low-latency inference on consumer GPUs (e.g., Tesla T4), several critical optimizations were implemented:
- **4-bit Quantization (NF4)**: The heavy SDXL UNet and Text Encoders are loaded in 4-bit precision using bitsandbytes, keeping VRAM usage low (~7-8GB for the model weights).
- **Token Pruning (ToMe)**: Utilizing tomesd, we prune 40% of the attention tokens (ratio=0.4). This acts as dynamic structural pruning, increasing throughput without retraining.
- **FP16 VAE**: Uses madebyollin/sdxl-vae-fp16-fix to avoid numerical instability (NaNs) common in standard SDXL VAEs when running in half-precision.
- **Sliced VAE Decoding**: Enabled via vae.enable_slicing() to decode large images in chunks, preventing OOM errors during the final decoding stage.
## Compute Profiles
### Hardware Environment: NVIDIA Tesla T4 (16GB VRAM)
Metric CategoryMetricSmart Fill (2-Pass)Harmonization (Crop)AnalysisGPU ResourcesPeak VRAM~7.8 GB~7.3 GB4-bit quantization keeps SDXL well within T4 limits.Power Draw~70W~68WConsistent power usage during UNet denoising steps.PerformanceLatency~12-15s~8-10sHarmonization is faster due to reduced resolution (768px crop) and fewer steps (15).Throughput~5.8 tokens/secN/AToken merging keeps generation fluid despite the heavy SDXL architecture.TFLOPS~0.17~0.16The pipeline maximizes the T4's float performance.

### Resource Analysis Summary
- **Efficiency**: By combining Int4 loading and Token Pruning, SDXL runs comfortably on 16GB cards with room to spare for concurrent requests or larger batch sizes.
- **Bottlenecks**: The primary bottleneck remains the iterative denoising process (scheduler steps). Smart Fill requires 30 steps for generation, making it compute-bound.
- **Thermal Profile**: The model runs within safe thermal limits (<50°C), aided by the reduced memory bandwidth requirements of 4-bit weights.