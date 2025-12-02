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
├── /diffusion_tobeused/
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

# Compute Performance Analysis: Harmonization vs. Generative Fill

## Executive Summary
This document compares the computational characteristics of **Harmonization** (Blue) and **Generative Fill** (Red). The data reveals two distinct workload profiles:
* **Generative Fill** is a hardware-saturating "stress test" that is bound by GPU power limits and memory bandwidth.
* **Harmonization** is a lighter, "bursty" workload that appears to be bound by I/O (Network/Disk) rather than raw compute.

---

## 1. Consolidated Compute Summary Table

| Metric Category | Metric Name | Harmonization (Blue) | Generative Fill (Red) | Insight / Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Duration** | `total_steps` | ~15 steps | ~30 steps | Generative Fill is exactly 2x the step count. |
| **Latency** | `latency_s` | ~22s $\to$ 21s | ~37s $\to$ 35s | Gen Fill is ~1.6x slower per step. |
| **GPU Load** | `GPU Utilization` | **Spiky** (Pulses 0-100%) | **Wall** (Solid 100%) | Harmonization allows the GPU to rest; Gen Fill pins it at max load. |
| **Power** | `GPU Power Usage` | Idles ~22W, spikes to 70W | Sustains **70W** (Limit) | Gen Fill effectively "brick walls" at the 70W hardware limit. |
| **Thermals** | `GPU Temperature` | Peaks **~34°C** | Peaks **~42°C** | Gen Fill heats the GPU significantly faster due to sustained load. |
| **Clock Speed** | `GPU SM Clock` | ~1580 MHz (stable) | ~1580 MHz (dips to 900) | Gen Fill forces the GPU to downclock (throttle) to stay under 70W. |
| **VRAM** | `GPU Mem Allocated` | ~7.4 GB (~45%) | ~9.0 GB (~56%) | Gen Fill requires significantly more VRAM headroom. |
| **Bandwidth** | `Time Accessing Mem`| <10% avg (spiky) | ~40-45% constant | Gen Fill is heavily bound by memory bandwidth. |
| **System RAM** | `Process Memory` | Drops to ~2.8 GB | Steady at ~3.2 GB | Harmonization active process memory is lower during the run. |
| **Disk I/O** | `Disk Read (MB)` | ~250 MB | ~200 MB | Harmonization reads more data from disk (likely model weights). |
| **Network** | `Network Traffic` | **High** (~17MB Recv) | **Low** (~9MB Recv) | Harmonization is 2x as chatty over the network, suggesting data streaming. |
| **Compute** | `Total TFLOPS` | ~0.045 | ~0.155 | Gen Fill is mathematically ~3.4x more expensive. |

---

## 2. Workload Profiles

### Generative Fill: "The Furnace"
* **Behavior:** This process acts as a stress test. It grabs every available resource and holds it until completion. It pins the GPU at 100% utilization and immediately hits the **70W Enforced Power Limit**.
* **Throttling:** Because it hits the power wall so hard, the GPU is forced to downclock (thermal/power throttle) from 1580 MHz to as low as 900 MHz to survive the workload.
* **Bottlenecks:**
    * **Power Bound:** Hard cap at 70W.
    * **Memory Bandwidth Bound:** It spends 45% of its time accessing memory.

### Harmonization: "The Sprinter"
* **Behavior:** This process works in short, intense bursts (spikes). It utilizes the GPU in pulses but allows it to rest and cool down in between.
* **Efficiency:** It is thermally efficient, peaking at only 34°C (vs 42°C for Gen Fill).
* **Bottlenecks:**
    * **I/O Bound:** It reads more data from the disk (250MB) and downloads significantly more data over the network (17MB). The GPU likely spends time waiting for this data, resulting in 0% utilization gaps.

---

## 3. Optimization Recommendations

1.  **To Speed Up Generative Fill:**
    * **Hardware Upgrade:** This workload is strictly limited by the GPU hardware. A GPU with a higher power limit (>70W) and higher memory bandwidth is required for faster inference.
    * **Cooling:** Ensure the device has adequate cooling, as the workload causes rapid thermal variance.

2.  **To Speed Up Harmonization:**
    * **I/O Optimization:** Focus on disk and network speeds. The GPU is underutilized. Moving model weights to faster NVMe storage or improving network download speeds for the input data will likely reduce the total runtime.

### Charts
![alt text](assets/Section-4-Panel-19-zi0m568sl.png) ![alt text](assets/Section-4-Panel-20-uy8a7yk1w.png) ![alt text](assets/Section-4-Panel-21-0w4dex6yu.png) ![alt text](assets/Section-2-Panel-0-qfho7iza6.png) ![alt text](assets/Section-2-Panel-1-w0lfq6r37.png) ![alt text](assets/Section-2-Panel-2-nnkqs6j8d.png) ![alt text](assets/Section-2-Panel-3-xxtm21cpo.png) ![alt text](assets/Section-2-Panel-4-byl7ls1rf.png) ![alt text](assets/Section-2-Panel-5-t42wqk4og.png) ![alt text](assets/Section-2-Panel-6-lquc9hls9.png) ![alt text](assets/Section-2-Panel-7-xyr9djojz.png) ![alt text](assets/Section-2-Panel-8-nm77dfanm.png) ![alt text](assets/Section-2-Panel-9-h4cl1t1tn.png) ![alt text](assets/Section-4-Panel-0-t9w5eqvok.png) ![alt text](assets/Section-4-Panel-1-8aucfv5zf.png) ![alt text](assets/Section-4-Panel-2-fif6xyq1o.png) ![alt text](assets/Section-4-Panel-3-9kcyle0jf.png) ![alt text](assets/Section-4-Panel-4-9hbgsom47.png) ![alt text](assets/Section-4-Panel-5-6fu4y3q94.png) ![alt text](assets/Section-4-Panel-6-0usjrj1gh.png) ![alt text](assets/Section-4-Panel-7-h47es1qad.png) ![alt text](assets/Section-4-Panel-8-qat4uz36p.png) ![alt text](assets/Section-4-Panel-9-ivlkr7616.png) ![alt text](assets/Section-4-Panel-10-buu7ktx3j.png) ![alt text](assets/Section-4-Panel-11-0v2rimcx0.png) ![alt text](assets/Section-4-Panel-12-2zhn685ri.png) ![alt text](assets/Section-4-Panel-13-eczzi41t8.png) ![alt text](assets/Section-4-Panel-14-6x1e4g9na.png) ![alt text](assets/Section-4-Panel-15-n4oqhxe9w.png) ![alt text](assets/Section-4-Panel-16-xjq0oyce6.png) ![alt text](assets/Section-4-Panel-17-wjfeuo292.png) ![alt text](assets/Section-4-Panel-18-oldxwnlb3.png)


### Resource Analysis Summary
- **Efficiency**: By combining Int4 loading and Token Pruning, SDXL runs comfortably on 16GB cards with room to spare for concurrent requests or larger batch sizes.
- **Bottlenecks**: The primary bottleneck remains the iterative denoising process (scheduler steps). Smart Fill requires 30 steps for generation, making it compute-bound.
- **Thermal Profile**: The model runs within safe thermal limits (<50°C), aided by the reduced memory bandwidth requirements of 4-bit weights.

## Citations

- [SDXL-inpainting model](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Controlnet-inpainting-dreamer](https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl)
- [SDXL-VAE (fp16)](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [(ToMe) Token Merging](https://huggingface.co/docs/diffusers/v0.29.1/optimization/tome)
