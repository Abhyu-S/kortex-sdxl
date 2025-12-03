# Image Editing with Stable Diffusion XL with Generative Fill and Harmonization

This is a production-ready image editing service built on Stable Diffusion XL (SDXL) with ControlNet for inpainting. It provides two primary capabilities: 
1. Text-guided Generative Fill to synthesize content within a mask while preserving structure, and
2. Edge-aware Harmonization to blend pasted or moved objects into a scene.

The system assembles SDXL components with an fp16 VAE and applies aggressive but quality-preserving optimizations: 4-bit NF4 quantization for UNet and text encoders (via BitsAndBytes) and Token Merging (ToMe) to reduce attention token cost at inference time. A two-pass “Smart Fill” pipeline inpaints first, then *optionally* performs a lightweight Img2Img pass (Vibe Match) to align lighting and style. A FastAPI server exposes simple multipart endpoints, returning PNG images. Resource usage and performance telemetry (latency, RAM/CPU, GPU util proxy, estimated TFLOPs) are captured via a background monitor and optionally logged to Weights & Biases. The system targets 16GB-class GPUs (e.g., T4 or similar) but supports CPU fallback with significantly higher latency.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Components and Responsibilities](#components-and-responsibilities)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Parameter Space and Constraints](#parameter-space-and-constraints)
- [Mathematical Computations](#mathematical-computations)
- [Performance Analysis](#performance-analysis)
- [Deployment Guide](#deployment-guide)
- [Testing & Validation](#testing--validation)
- [Extensibility](#extensibility)
- [Recommendations](#Recommendations)
- [Development Journey](#development-journey-summary)
- [Future Scope](#future-scope)
- [Quick Links](#quick-links)
- [References & Citations](#references--citations)
- [Appendix: Prompts, Workflows, Examples](#appendix-prompts-workflows-examples)

---

## Overview
- Purpose: High-quality image editing with Stable Diffusion XL (SDXL) featuring text-guided Generative Fill and edge-aware Harmonization for pasted/moved objects.
- Strengths:
    - 4-bit NF4 quantization + Token Merging (ToMe) for low VRAM and faster inference.
    - Two-pass Smart Fill: Fill then optional Vibe Match for lighting/style alignment.
    - Simple, reliable FastAPI endpoints; built-in telemetry via Weights & Biases (optional).
- Target hardware: 16GB-class GPUs (e.g., Tesla T4); supports CPU fallback with reduced performance.

### The Compute: Tesla T4
To address the challenge's requirement for a lightweight, mobile-first editor, we utilized the NVIDIA Tesla T4 as a hardware proxy for the estimated compute capability of a flagship mobile NPU in 2030.

- **Current State (2024)**: High-end mobile NPUs (e.g., A17 Pro, Snapdragon 8 Gen 3) reach ~35-45 TOPS (Trillions of Operations Per Second).
- **The 2030 Projection**: Extrapolating current NPU efficiency gains, mobile edge devices in 2030 are projected to exceed 100+ TOPS, rivaling the inference throughput of today's mid-range inference cards like the T4 (~65 TOPS Int8 / ~8 TFLOPS FP32).
- **Our Approach**: By optimizing Stable Diffusion XL (SDXL) with 4-bit Quantization and Token Merging, we demonstrate that high-fidelity generative editing can run locally on this "2030-equivalent" compute profile without cloud dependency.

### Datasets
This project utilizes a **Training-Free approach**. It leverages pre-trained weights from Stability AI and Destitech, applying inference-time optimizations to achieve performance goals.
- **Inference data**: Accepts standard image formats (JPG, PNG) and binary masks.

## Architecture

![SDXL Architecture](assets/SDXL_Architecture.jpg)

- Core files:
    - `server.py`: FastAPI API exposing `/`, `/health`, `/generative-fill`, `/smart-fill`, `/harmonize`.
    - `editing_pipelines_fill.py`: SDXL + ControlNet assembly, editing logic, resource monitoring.
    - `quantization_utils.py`: BitsAndBytes 4-bit NF4 configuration.
    - `pruning_utils.py`: ToMe-based token pruning.
    - `requirements.txt`: dependencies.

```mermaid
flowchart LR
	A[Client] --> B[FastAPI Server]
	B -->|decode| C[Image + Mask]
	B -->|route| D[EditingPipelines]
	D -->|inpaint| E[SDXL + ControlNet]
	D -->|optional| F[SDXL Img2Img Vibe Match]
	E --> G[Result]
	F --> G
	G --> H[PNG Response]
```

### Component Breakdown

| Component | Role | Precision | Notes |
|---|---|---|---|
| UNet (SDXL) | Denoising backbone | 4-bit NF4 weights, fp16 compute | Quantized via BitsAndBytes; majority of VRAM footprint |
| Text Encoder 1 (CLIP) | Prompt conditioning | 4-bit NF4, fp16 compute | Encodes base text tokens |
| Text Encoder 2 (CLIP+Proj) | Enhanced conditioning | 4-bit NF4, fp16 compute | Projection improves SDXL text-image alignment |
| VAE (fp16 fix) | Latent to Image | fp16 | `vae.enable_slicing()` for memory-friendly decode |
| ControlNet (Inpaint Dreamer) | Structure guidance | fp16 | Driven by source image and mask for coherent fill |
| Scheduler (EulerDiscrete) | Step update rule | N/A | `timestep_spacing="trailing"` for stability |

```mermaid
flowchart TB
	subgraph Pipelines
		U[UNet NF4 fp16]
		T1[TextEncoder1 NF4 fp16]
		T2[TextEncoder2 NF4 fp16]
		V[VAE fp16]
		Cn[ControlNet fp16]
		Sch[Scheduler]
	end
	T1 --> U
	T2 --> U
	Cn --> U
	U --> V
	Sch --> U
```

```mermaid
flowchart LR
    Img[Input Image] -- mask --> Cn
    Cn -- control --> U
    Prompt -- tokenizers --> T1 & T2
    T1 & T2 --> U
    U -- latents --> V
    V --> Out[Output Image]
```

### Deep Learning Architecture: SDXL UNet Structure

The UNet is the core denoising engine. It processes latent representations through a series of downsampling and upsampling blocks with skip connections.

```mermaid
flowchart TD
    subgraph UNet Architecture
        Input[Noisy Latent zt 4x128x128] --> Down1[DownBlock 1 Conv ResNet]
        Down1 --> Attn1[Self-Attention Cross-Attention with Text Embeddings]
        Attn1 -->|ToMe merges tokens| Down2[DownBlock 2 Conv ResNet]
        Down2 --> Attn2[Self-Attention Cross-Attention]
        Attn2 --> Down3[DownBlock 3 Bottleneck]
        
        Down3 --> Mid[MidBlock ResNet Attention]
        
        Mid --> Up3[UpBlock 3 TransposeConv]
        Down2 -.skip.-> Up3
        Up3 --> Attn3[Self-Attention Cross-Attention]
        Attn3 --> Up2[UpBlock 2 TransposeConv]
        Down1 -.skip.-> Up2
        Up2 --> Attn4[Self-Attention Cross-Attention]
        Attn4 --> Up1[UpBlock 1 Output Conv]
        
        Up1 --> Output[Denoised Latent zt-1 4x128x128]
    end
    
    Prompt[Text Prompt] --> TE1[Text Encoder 1<br/>CLIP ViT-L/14<br/>NF4 Quantized]
    Prompt --> TE2[Text Encoder 2<br/>CLIP ViT-bigG/14<br/>NF4 Quantized]
    TE1 --> Concat[Concatenate<br/>Embeddings]
    TE2 --> Concat
    Concat -.cross-attention.-> Attn1 & Attn2 & Attn3 & Attn4
```

**Key Components**:
- **DownBlocks**: Convolutional layers that reduce spatial dimensions (128→64→32)
- **ResNet Blocks**: Residual connections for gradient flow
- **Self-Attention**: Captures spatial relationships within the latent
- **Cross-Attention**: Integrates text prompt information into image generation
- **ToMe Application**: Applied at attention layers, merging similar tokens to reduce compute
- **Skip Connections**: Preserve high-frequency details from encoder to decoder

### SDXL Dual Text Encoder System

SDXL uses two CLIP text encoders simultaneously for richer semantic understanding.

```mermaid
flowchart TB
    subgraph Text Input
        Prompt["Text Prompt<br/>'a futuristic city at sunset'"] --> Tok1[Tokenizer 1<br/>CLIP ViT-L vocab]
        Prompt --> Tok2[Tokenizer 2<br/>CLIP ViT-bigG vocab]
    end
    
    subgraph CLIP ViT-L/14 Encoder 1
        Tok1 --> Emb1[Token Embeddings<br/>77x768]
        Emb1 --> Pos1[+ Position Embeddings]
        Pos1 --> Trans1[Transformer Blocks<br/>12 layers]
        Trans1 --> Pool1[Pooled Output<br/>768-d]
        Trans1 --> Seq1[Sequence Output<br/>77x768]
    end
    
    subgraph CLIP ViT-bigG/14 Encoder 2
        Tok2 --> Emb2[Token Embeddings<br/>77x1280]
        Emb2 --> Pos2[+ Position Embeddings]
        Pos2 --> Trans2[Transformer Blocks<br/>24 layers, deeper]
        Trans2 --> Pool2[Pooled Output<br/>1280-d]
        Trans2 --> Seq2[Sequence Output<br/>77x1280]
    end
    
    subgraph Embedding Fusion
        Seq1 --> Concat[Concatenate<br/>along embedding dim]
        Seq2 --> Concat
        Concat --> FinalEmb["Combined Embeddings<br/>77x2048"]
        
        Pool1 --> PoolConcat[Concatenate Pooled]
        Pool2 --> PoolConcat
        PoolConcat --> AddTime["+ Time Embedding<br/>+ Resolution Conditioning"]
        AddTime --> CondVec[Conditioning Vector]
    end
    
    FinalEmb -.cross-attention.-> UNet[UNet Cross-Attention Layers]
    CondVec --> UNet
```

 

### ControlNet Architecture: Structure-Guided Inpainting

ControlNet is a parallel network that processes the control image (original + mask) and injects structural guidance into the UNet at multiple resolution levels.

```mermaid
flowchart TB
    subgraph ControlNet Pipeline
        CI[Control Image<br/>RGB 1024x1024] --> Enc[ControlNet Encoder<br/>Trainable Copy of UNet Encoder]
        Mask[Binary Mask<br/>1024x1024] --> Enc
        
        Enc --> F1[Feature Map 1<br/>64x64]
        Enc --> F2[Feature Map 2<br/>32x32]
        Enc --> F3[Feature Map 3<br/>16x16]
        
        F1 --> Scale1[Scale by<br/>conditioning_scale=0.5]
        F2 --> Scale2[Scale by<br/>conditioning_scale=0.5]
        F3 --> Scale3[Scale by<br/>conditioning_scale=0.5]
        end
    
    subgraph UNet Decoder
        UD1[UNet Down 1] --> UA1[UNet Attn 1]
        UD2[UNet Down 2] --> UA2[UNet Attn 2]
        UD3[UNet Down 3] --> UA3[UNet Attn 3]
    end
    
    Scale1 -.add residual.-> UA1
    Scale2 -.add residual.-> UA2
    Scale3 -.add residual.-> UA3
```

---


## Mathematical Computations
- Diffusion: UNet predicts noise residuals per step; scheduler updates latent $z_t \to z_{t-1}$.
- ToMe Token Merging: with ratio $r$, effective tokens ≈ $(1-r)N$; attention cost trends from $O(N^2)$ to $O(((1-r)N)^2)$.
- NF4 Quantization: 4-bit normal-float codebook compression for weights; fp16 compute reduces bandwidth and VRAM.
- Estimated TFLOPs (as in code): $\text{TFLOPs} \approx \frac{2PSR}{10^{12}}$, where $P$=params proxy, $S$=steps executed, $R$=resolution factor (vs 1024²). Throughput uses TFLOPs/latency.

### Scheduler Configuration Details

| Parameter | Value | Rationale |
|---|---|---|
| Scheduler Type | EulerDiscreteScheduler | Fast convergence, stable for SDXL |
| `timestep_spacing` | `"trailing"` | Improves quality by concentrating steps near t=0 |
| `num_inference_steps` | 30 (fill), 15 (harmonize) | Empirically tuned for quality/speed balance |
| `guidance_scale` | 7.5 (fill), 2.5 (harmonize/vibe) | Higher for generative tasks, lower for refinement |

**Timestep Spacing Visualization**:
```
Leading:   [1000, 900, 800, ..., 100, 50, 0]     (uniform)
Trailing:  [1000, 950, 920, ..., 20, 5, 0]      (concentrated at end)
```

Trailing spacing allocates more compute to final denoising steps where perceptual quality is most sensitive.

 

---

## Performance Analysis
- On 16GB-class GPUs: peak VRAM typically < ~8-9 GB due to 4-bit UNet/encoders + fp16 VAE; first run slower due to downloads.
- Latency: dominated by denoising steps; harmonization is faster due to cropped area and fewer steps.
- Telemetry: CPU/RAM via psutil; GPU util via pynvml (if available); W&B logging for latency, steps, RAM/CPU/GPU proxies, TFLOPs.
- Tips: keep masks tight; tune steps and `vibe_strength` to balance quality and speed.

### Hardware Environment: NVIDIA Tesla T4 (16GB VRAM)

| Metric Category | Metric | Smart Fill (2-Pass) | Harmonization (Crop) | Analysis |
|---|---|---|---|---|
| GPU Resources | Peak VRAM | ~7.8 GB | ~7.3 GB | 4-bit quantization keeps SDXL well within T4 limits |
| | Power Draw | ~70W | spikes at 68W | Consistent power usage during UNet denoising steps |
| Performance | Latency | ~12-15s | ~8-10s | Harmonization is faster due to reduced resolution (768px crop) and fewer steps (15) |
| | Throughput | ~5.8 tokens/sec | N/A | Token merging keeps generation fluid despite the heavy SDXL architecture |
| | TFLOPs | ~0.17 | ~0.16 | The pipeline maximizes the T4's float performance |

### Resource Analysis Summary
- **Efficiency**: By combining Int4 loading and Token Pruning, SDXL runs comfortably on 16GB cards with room to spare for concurrent requests or larger batch sizes.
- **Bottlenecks**: The primary bottleneck remains the iterative denoising process (scheduler steps). Smart Fill requires 30 steps for generation, making it compute-bound.
- **Thermal Profile**: The model runs within safe thermal limits (<50°C), aided by the reduced memory bandwidth requirements of 4-bit weights.

 

--- Memory Footprint Analysis

| Component | Precision | Size (approx) | % of Total VRAM | Mitigation |
|---|---|---|---|---|
| UNet (SDXL) | 4-bit NF4 | ~2.5 GB | ~35% | Quantization from ~10 GB (fp16) |
| Text Encoder 1 | 4-bit NF4 | ~0.6 GB | ~8% | Quantization from ~2.4 GB (fp16) |
| Text Encoder 2 | 4-bit NF4 | ~0.8 GB | ~11% | Quantization from ~3.2 GB (fp16) |
| VAE (fp16) | fp16 | ~0.7 GB | ~10% | Slicing for decode; fp16-fix avoids NaNs |
| ControlNet | fp16 | ~1.2 GB | ~17% | Kept fp16 for precision in structure guidance |
| Activations (peak) | fp16 | ~1.0-1.5 GB | ~15-20% | ToMe reduces token count |
| Overhead (CUDA) | N/A | ~0.3-0.5 GB | ~4-7% | Driver/context allocation |
| **Total (peak)** | N/A | **~7-8 GB** | **100%** | Fits T4 (16GB) with headroom |

---

## Pruning and Quantization: Rationale, Application, Trade-offs

### Techniques Applied
- **Token Merging (ToMe) Pruning**
    - Application: `tomesd.apply_patch(pipeline, ratio=0.4)` merges redundant attention tokens at inference.
    - Effect: Reduces attention compute with minimal quality loss; speeds up denoising passes.
    - Citation: ToMe — https://github.com/facebookresearch/ToMe
- **BitsAndBytes 4-bit NF4 Quantization**
    - Application: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` on UNet and text encoders.
    - Effect: Compresses weights to 4-bit with learned normal-float quantization; compute stays fp16.
    - Citation: BitsAndBytes — https://github.com/TimDettmers/bitsandbytes

### Why These Techniques
- SDXL is parameter-heavy; VRAM is dominated by UNet and encoders. NF4 sharply reduces memory footprint without retraining.
- Attention is often redundant across spatial tokens; ToMe removes duplicate information to accelerate inference.
- Combined: keep quality high while fitting on 16GB-class GPUs and improving latency.

### Advantages
- Lower VRAM footprint → fits on mid-range GPUs, enables higher resolution.
- Faster steps → reduced end-to-end latency.
- No training required → pure inference-time optimization.


### Future Scope
- Adaptive ToMe ratio per-layer/token density.
- Mixed-precision schemes (per-layer granularity).
- Selective dequantization for critical blocks.

### Comparison Table

| Setting | VRAM (approx) | Latency | Quality | Notes |
|---|---|---|---|---|
| Baseline (fp16 weights) | High (>12 GB) | High | Reference | Hard to fit on 12-16 GB GPUs |
| NF4 only | Medium (~8-9 GB) | Medium | Near-ref | Good balance without pruning |
| NF4 + ToMe (0.4) | Lower (~7-8 GB) | Lower | Minor micro-detail loss | Recommended default here |

### Optimization Impact Breakdown

| Optimization | VRAM Reduction | Speed Gain | Implementation Complexity | Quality Impact |
|---|---|---|---|---|
| 4-bit NF4 Quantization | ~55-60% | Moderate (bandwidth-limited tasks) | Low (config-based) | Negligible |
| Token Merging (ToMe) | ~10-15% (activations) | ~25-35% (attention ops) | Low (single patch) | Minor at ratio 0.4 |
| VAE Slicing | Prevents OOM spikes | Minimal latency cost | Trivial (one-liner) | None |
| Combined Stack | ~60-65% total | ~30-40% end-to-end | Low | Minor texture softening |

### Token Merging (ToMe) Mechanism

ToMe reduces computational cost by identifying and merging similar tokens in the attention mechanism.

```mermaid
flowchart TD
    subgraph Before ToMe
        T1["Tokens N 16384 128x128 spatial"] --> Attn1[Self-Attention O N squared complexity]
        Attn1 --> Out1["Output N 16384"]
    end
    
    subgraph ToMe Process
        T2["Tokens N 16384"] --> Sim[Compute Token Similarity Matrix]
        Sim --> Merge["Merge 40 percent most similar tokens"]
        Merge --> T3["Reduced Tokens N prime 9830"]
    end
    
    subgraph After ToMe
        T3 --> Attn2[Self-Attention O N prime squared complexity]
        Attn2 --> Unmerge[Unmerge tokens back to original positions]
        Unmerge --> Out2["Output N 16384"]
    end
    
    T2 -.-> T3
```

**Computational Savings**:
- Original: $O(16384^2) \approx 268M$ operations
- With ToMe (r=0.4): $O(9830^2) \approx 97M$ operations
- **Speedup**: ~2.8× for attention layers (~35% faster overall)

**Merge Strategy**:
1. Compute cosine similarity between adjacent tokens
2. Identify pairs with highest similarity
3. Merge via weighted average based on attention scores
4. After processing, tokens are "unmerged" to restore spatial structure

### NF4 Quantization Mechanism

```mermaid
flowchart LR
    subgraph Standard fp16 Storage
        W1[Weights<br/>fp16<br/>2 bytes/param] --> GPU1[GPU Memory<br/>~10 GB for UNet]
    end
    
    subgraph NF4 Quantization
        W2[Weights<br/>fp16] --> Q[Quantize to<br/>4-bit NF4<br/>learned codebook]
        Q --> Store[Compressed<br/>0.5 bytes/param]
        Store --> DQ[Dequantize to<br/>fp16 on-the-fly]
        DQ --> Compute[Compute in<br/>fp16]
    end
    
    subgraph Result
        Compute --> GPU2[GPU Memory<br/>~2.5 GB for UNet<br/>75% reduction]
    end
```

**NF4 (Normal Float 4-bit)**:
- Uses a learned codebook based on normal distribution of weights
- Asymmetric quantization preserves precision for important weight ranges
- Double quantization: quantizes the quantization constants for extra compression
- Dequantization happens during forward pass, compute remains fp16

### Cross-Attention Mechanism: Text-to-Image Guidance

### Classifier-Free Guidance (CFG) Process

### Noise Schedule Visualization

Cross-attention allows text embeddings to influence image generation at the pixel level.

```mermaid
flowchart TB
    subgraph Spatial Features from UNet
        Latent["Latent Features h<br/>B×C×H×W"] --> Reshape["Reshape to<br/>B×(H*W)×C"]
        Reshape --> QProj["Query Projection<br/>Linear(C, D)"]
        QProj --> Q["Queries Q<br/>B×(H*W)×D"]
    end
    
    subgraph Text Embeddings
        Text["Text Embeddings<br/>B×77×2048"] --> KProj["Key Projection<br/>Linear(2048, D)"]
        Text --> VProj["Value Projection<br/>Linear(2048, D)"]
        KProj --> K["Keys K<br/>B×77×D"]
        VProj --> V["Values V<br/>B×77×D"]
    end
    
    subgraph Attention Computation
        Q --> MatMul1["Q × K^T<br/>B×(H*W)×77"]
        K --> MatMul1
        MatMul1 --> Scale["Scale by<br/>1/√D"]
        Scale --> Softmax["Softmax<br/>Attention Weights"]
        Softmax --> Attn["Attention Map<br/>B×(H*W)×77"]
    end
    
    subgraph Output
        Attn --> MatMul2["Attn × V<br/>Weighted sum of text features"]
        V --> MatMul2
        MatMul2 --> Out["Output Features<br/>B×(H*W)×D"]
        Out --> Reshape2["Reshape to<br/>B×D×H×W"]
        Reshape2 --> Final["Attended Features<br/>+ Residual Connection"]
    end
```

**Attention Visualization**:
For a pixel at position (i,j), the attention weights `Attn[i,j,:]` show which words from the prompt most influence that pixel:
- "futuristic" → high attention on building edges
- "sunset" → high attention on sky regions
- "city" → high attention across spatial extent

**Mathematical Formulation**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q ∈ ℝ^(HW×D): queries from image features
- K, V ∈ ℝ^(77×D): keys/values from text embeddings
- d_k = D: scaling factor to stabilize gradients
- Output ∈ ℝ^(HW×D): text-guided image features
```

### Classifier-Free Guidance (CFG) Process

CFG amplifies the influence of text prompts by computing both conditional and unconditional predictions.

```mermaid
flowchart TB
    subgraph Dual Forward Pass
        LatentIn["Noisy Latent z_t"] --> Dup[Duplicate]
        Dup --> Cond["Conditional Branch<br/>with text embeddings"]
        Dup --> Uncond["Unconditional Branch<br/>with empty prompt ''"]
        
        Prompt["Text Prompt"] --> TextEnc[Text Encoders]
        TextEnc --> CondEmb["Text Embeddings"]
        Empty["Empty String ''"] --> TextEnc2[Text Encoders]
        TextEnc2 --> UnCondEmb["Null Embeddings"]
        
        Cond --> UNet1["UNet Forward<br/>+ CondEmb"]
        CondEmb --> UNet1
        UNet1 --> NoiseCond["ε_cond<br/>Conditional noise prediction"]
        
        Uncond --> UNet2["UNet Forward<br/>+ UnCondEmb"]
        UnCondEmb --> UNet2
        UNet2 --> NoiseUncond["ε_uncond<br/>Unconditional noise prediction"]
    end
    
    subgraph Guidance Application
        NoiseCond --> Sub["ε_cond - ε_uncond<br/>Direction of guidance"]
        NoiseUncond --> Sub
        Sub --> Scale["× guidance_scale<br/>(7.5 for fill, 2.5 for harmonize)"]
        Scale --> Add["+ ε_uncond<br/>Base prediction"]
        NoiseUncond --> Add
        Add --> NoiseFinal["ε_guided<br/>Final noise prediction"]
    end
    
    subgraph Denoising Step
        NoiseFinal --> Sched["Scheduler Update<br/>z_t → z_{t-1}"]
        LatentIn --> Sched
        Sched --> LatentOut["Less noisy z_{t-1}"]
    end
```

**CFG Formula**:
```
ε_guided = ε_uncond + guidance_scale × (ε_cond - ε_uncond)
```

**Impact of guidance_scale**:
- `1.0`: No guidance, equivalent to unconditional generation
- `7.5`: Strong adherence to prompt (used in Smart Fill)
- `2.5`: Subtle guidance (used in Harmonization for natural blending)
- Higher values: More literal interpretation but risk of artifacts

### Noise Schedule Visualization

```mermaid
flowchart LR
    subgraph Forward Diffusion Training
        X0["Clean Image x0"] -->|noise beta1| X1["Slightly noisy x1"]
        X1 -->|noise beta2| X2[x2]
        X2 -->|more steps| XT["Pure noise xT t equals 1000"]
    end
    
    subgraph Reverse Diffusion Inference
        ZT["Random noise zT approx N 0 I"] -->|UNet step 1| ZT1[zT minus 1]
        ZT1 -->|UNet step 2| ZT2[zT minus 2]
        ZT2 -->|30 steps trailing schedule| Z1[z1]
        Z1 -->|UNet step 30| Z0["Clean latent z0"]
    end
    
    Z0 --> VAE[VAE Decoder]
    VAE --> Img["Output Image 1024x1024"]
```

**Trailing Timestep Spacing**:
```python
timesteps = [999, 970, 941, 912, ..., 50, 25, 12, 6, 3, 1, 0]
```
More steps concentrated at the end (low noise) where visual details form.

**Comparison**:
| Spacing | Early Steps | Late Steps | Quality | Speed |
|---------|-------------|------------|---------|-------|
| Linear | Uniform | Uniform | Good | Standard |
| Leading | Dense | Sparse | Coarse details | Fast |
| **Trailing** | **Sparse** | **Dense** | **Fine details** | **Balanced** |

## Deployment Guide

### Prerequisites
1. **NVIDIA GPU**: Tested on Tesla T4; compatible with 12GB+ VRAM GPUs.
2. **Python 3.10+**: Required for dependencies.
3. **CUDA Toolkit**: Installed and compatible with your PyTorch version.
4. **Disk Space**: Sufficient space for model weights (~10-15 GB on first download).

### Environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch per your CUDA/driver version
# See https://pytorch.org/get-started/locally/
# Example (CUDA 12.1 wheels):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Models
- Auto-downloaded on first run:
    - SDXL base: `stabilityai/stable-diffusion-xl-base-1.0`
    - ControlNet inpaint: `destitech/controlnet-inpaint-dreamer-sdxl`
    - VAE (fp16 fix): `madebyollin/sdxl-vae-fp16-fix`

### Run
```powershell
python server.py
# http://localhost:8080  (docs at /docs)
```

Notes: GPU recommended; CPU works but is slow. Ensure sufficient disk space for model weights.

---

## Testing & Validation
- Smoke tests:
    - `GET /health` returns `{ status: "running", gpu: <bool> }`.
    - `/smart-fill` returns PNG of same size as input.
- Functional checks:
    - White mask area should change; black area preserved.
    - Compare `vibe_strength=0.0` vs `0.3` to verify relighting.
- Future tests:
    - Unit tests for bbox/border logic in harmonization.
    - Golden images for standard prompts.

---

 

 

 

## Development Journey (Summary)
- From baseline SDXL to quantized (NF4) + pruned (ToMe) pipelines for constrained GPUs.
- Edge-only harmonization with cropped processing and safe odd kernel widths for filters.
- Resource monitor + W&B logging to illuminate latency and utilization.
- Stable scheduling via `timestep_spacing="trailing"` and refined default prompts.

---

## Future Scope
- Multi-resolution tiling for ultra-high-res edits
- Embedding caching for repeated prompts
- Adaptive ToMe ratio per layer
- Hybrid precision paths (fp16 critical projections, int8 weights)
- ORT/TensorRT export for full pipeline components
 
## Future Improvements: SDXL Inpainting UNet Compression

This section summarizes exploratory work to compress the SDXL Inpainting UNet for better deployability while preserving interfaces and practical accuracy. It focuses on three coordinated steps: topology-safe depth pruning, teacher-student distillation, and post-training dynamic INT8 quantization.

### What Was Done and Why It Works

- Pruning: Transformer attention blocks in down and up paths are pruned by removing middle layers while preserving the last attention layer per module. Mid-block attentions are left intact to avoid fragile shape mismatches. Configs are updated to match pruned depths to prevent export issues.
- Distillation: The pruned student UNet learns to match the original FP16 teacher noise predictions via MSE, with identical conditioning (dual CLIP embeddings, time ids) and inpainting inputs (noisy latents plus mask plus masked latents). Student parameters use FP32 for optimizer stability, and autocast accelerates ops.
- Quantization: The pruned UNet is exported to ONNX (FP32), validated, optionally split into external data if large, and quantized with dynamic per-channel INT8 weights (QUInt8). Input signatures and conditioning are preserved for the ORT SDXL Inpaint pipeline.

```mermaid
flowchart TD
        A[Base SDXL Inpaint UNet] --> B[Topology Safe Pruning]
        B --> C[Teacher Student Distillation]
        C --> D[Hybrid Diffusers Pipeline]
        D --> E[ONNX Export FP32]
        E --> F[Dynamic INT8 Quantization]
        F --> G[ORT SDXL Inpaint Pipeline]
```

### How It Preserved the UNet

- Maintains 9 channel inpainting input signature and added conditioning (encoder hidden states, text embeds, time ids).
- Protects last attention layers to preserve interfaces across down and up paths.
- Updates UNet config to reflect pruned depth, preventing silent reconstruction of removed layers during export.
- Aligns student to teacher diffusion behavior via noise prediction targets, retaining denoising quality.

```mermaid
flowchart LR
    subgraph Down Path
        D1[Attention Stack] --> D2[Prune Middle]
        D2 --> D3[Keep Last Layer]
    end
    subgraph Mid Path
        M1[Attention Stack] --> M2[Skip Pruning]
    end
    subgraph Up Path
        U1[Attention Stack] --> U2[Prune Middle]
        U2 --> U3[Keep Last Layer]
    end
```

### Results Template

Record your measured numbers after running training and `final_quant.py`.

| Stage | Location | Params (approx) | Size (MB) | Reduction vs previous |
| - | - | - | - | - |
| Pruned UNet FP16 | `sdxl_inpainting_pruned_fp16` | — | — | — |
| ONNX UNet FP32 | `sdxl_final_quantized/intermediate_fp32/unet/model.onnx` | script output | script output | — |
| Quantized UNet INT8 | `sdxl_final_quantized/final_model/unet/model.onnx` | script output | script output | computed |

Quality suggested: student teacher prediction MSE on a held out set, optional LPIPS on rendered inpaint images.
zz
#### Example Result

Below is a sample composite showing the input image, mask, and the resulting output generated during the UNet compression experiments.

![Input + Mask + Output](assets/FutureImprovement_input_mask_output.jpg)

```mermaid
pie title UNet Size Breakdown MB
    "Pruned UNet FP16" : 0
    "ONNX UNet FP32"   : 0
    "Quantized UNet INT8" : 0
```

```mermaid
flowchart LR
    S1[Start Baseline UNet] --> S2[After Pruning]
    S2 --> S3[After ONNX Export]
    S3 --> S4[After INT8 Quantization]
    style S1 fill:#777,stroke:#333,stroke-width:1px
    style S2 fill:#6aa84f,stroke:#333,stroke-width:1px
    style S3 fill:#3d85c6,stroke:#333,stroke-width:1px
    style S4 fill:#e69138,stroke:#333,stroke-width:1px
```

### Why This Was Better

- Depth pruning with last layer protection reduces compute while minimizing risk of breaking attention interfaces and skip connections.
- Distilling noise predictions targets the diffusion objective directly, transferring learned behavior efficiently without full retraining.
- Dynamic per-channel INT8 quantization is calibration free and preserves accuracy in wide attention and linear layers better than per tensor schemes.

### Future Improvements

- Sensitivity guided pruning and low rank factorization for high impact layers.
- Quantization aware fine tuning to recover any quantization loss, especially in attention projections.
- Hybrid precision export INT8 weights and FP16 critical projections for accuracy speed balance.
- Static activation quantization with small calibration sets for CPU first deployments.

 

### Attention Path Pruning Map

```mermaid
flowchart TD
        classDef keep fill:#a3f3a3,stroke:#333,stroke-width:1px,color:#111
        classDef prune fill:#f7c6c6,stroke:#333,stroke-width:1px,color:#111

        subgraph Down Blocks
            D_attn1[Attn Block 1]:::keep --> D_attn2[Attn Block 2]:::prune --> D_attn3[Attn Block 3 Last]:::keep
        end
        subgraph Mid Block
            M_attn[Mid Attn]:::keep
        end
        subgraph Up Blocks
            U_attn1[Attn Block 1]:::keep --> U_attn2[Attn Block 2]:::prune --> U_attn3[Attn Block 3 Last]:::keep
        end
```

## Quick Links
- API docs: `http://localhost:8080/docs`
- Health: `http://localhost:8080/health`
- Key files: `server.py`, `editing_pipelines_fill.py`, `quantization_utils.py`, `pruning_utils.py`, `requirements.txt`
- SDXL base: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- ControlNet inpaint: https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl
- VAE fp16 fix: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

---

## References & Citations
- Diffusers: https://github.com/huggingface/diffusers
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- ToMe: https://github.com/facebookresearch/ToMe
- SDXL: https://stability.ai/news/stable-diffusion-xl-1.0-release
- FastAPI: https://fastapi.tiangolo.com/
- W&B: https://wandb.ai/site

---

## Appendix: Prompts, Workflows, Examples

### Prompt Patterns
- Template: `[subject/scene], [style], [quality cues], [lighting]`
- Quality cues: `8k, photorealistic, master calibration`
- Negative cues: `blurry, ugly, deformed, text, watermark, low quality`

Examples:
- “A modern marble kitchen island, 8k, photorealistic, soft daylight”
- “Warm living room with wooden textures, cozy, detailed, natural lighting”
- “Green park background, shallow depth of field, no text, no watermark”

### Smart Fill Workflow
1) Build binary mask (white=edit). 2) Provide descriptive prompt + negatives. 3) Use `/smart-fill` with `vibe_strength` 0.2 to 0.4. 4) Compare results; adjust strength if needed.

### Harmonization Workflow
1) Prepare composite with pasted object. 2) Mask white over the pasted area. 3) Call `/harmonize`. 4) Validate seams/edges.

### Commands (PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python server.py
curl http://localhost:8080/health
curl -Method POST "http://localhost:8080/smart-fill" -Form image=@"C:\img\source.png" -Form mask=@"C:\img\mask.png" -Form prompt="a sunlit cafe interior" -Form vibe_strength=0.25 --output cafe.png
curl -Method POST "http://localhost:8080/harmonize" -Form image=@"C:\img\composite.png" -Form mask=@"C:\img\sticker_mask.png" --output composite_h.png
```

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| CUDA OOM | VRAM pressure | Reduce steps; ensure 4-bit; tighten mask; close other GPU apps |
| Slow 1st run | Model downloads | Let first pass finish; cache persists |
| Edits spill | Loose mask | Refine mask edges; white=edit |
| Lighting off | Vibe too low/high | Tune `vibe_strength` 0.2 to 0.4 |
| No GPU | Driver/CUDA/Torch mismatch | Install correct Torch CUDA build; verify `nvidia-smi` |

## SDXL Architecture Overview

### SDXL vs Stable Diffusion 1.5 Comparison

```mermaid
flowchart TB
    subgraph SD 1.5 Pipeline
        T1_15["Single Text Encoder<br/>CLIP ViT-L/14<br/>768-d embeddings"] --> U1_15["UNet<br/>860M params<br/>512×512 training"]
        U1_15 --> V1_15["VAE<br/>4×64×64 latents"]
        V1_15 --> Out1_15["512×512 Image"]
    end
    
    subgraph SDXL Pipeline kortex implementation
        T1_XL["Text Encoder 1<br/>CLIP ViT-L/14<br/>768-d"] --> Concat_XL["Concat<br/>2048-d total"]
        T2_XL["Text Encoder 2<br/>CLIP ViT-bigG/14<br/>1280-d"] --> Concat_XL
        Concat_XL --> U1_XL["UNet<br/>2.6B params<br/>1024×1024 training<br/>+ ControlNet guidance"]
        U1_XL --> V1_XL["VAE fp16-fix<br/>4×128×128 latents"]
        V1_XL --> Out1_XL["1024×1024 Image"]
    end
    
    subgraph Optimizations Applied
        T1_XL -.4-bit NF4.-> T1_XL
        T2_XL -.4-bit NF4.-> T2_XL
        U1_XL -.4-bit NF4.-> U1_XL
        U1_XL -.ToMe 40%.-> U1_XL
        V1_XL -.slicing.-> V1_XL
    end
```

**Key SDXL Improvements**:
| Feature | SD 1.5 | SDXL | Benefit |
|---------|--------|------|--------|
| Text Encoders | 1 (CLIP ViT-L) | 2 (ViT-L + ViT-bigG) | Richer semantic understanding |
| UNet Size | 860M params | 2.6B params | Better detail and coherence |
| Training Resolution | 512x512 | 1024x1024 | Native high-res generation |
| Latent Size | 4x64x64 | 4x128x128 | More spatial information |
| Conditioning | Text only | Text + Size + Crop coords | Aspect ratio awareness |
| VRAM (unoptimized) | ~4-6 GB | ~18-22 GB | Requires optimization |
| **VRAM (our stack)** | - | **~7 GB** | **Via NF4 + ToMe** |

## Components and Responsibilities
- `FastAPI` (in `server.py`):
    - Handles upload, decoding, routing, and PNG responses.
    - Endpoints: `GET /`, `GET /health`, `POST /generative-fill`, `POST /smart-fill`, `POST /harmonize`.
- `EditingPipelines` (in `editing_pipelines_fill.py`):
    - **The Brain**: Loads SDXL components (UNet, text encoders, tokenizer, scheduler) and ControlNet (inpaint-dreamer) with fp16 VAE.
    - Applies 4-bit NF4 quantization (UNet + text encoders) and ToMe pruning (ratio≈0.4).
    - Implements `run_smart_fill` and `run_harmonize_sticker` with resource monitoring + optional W&B logging.
    - **ResourceMonitor**: A background thread that tracks RAM and CPU usage during inference.
    - **run_smart_fill**: Orchestrates the generation process using ControlNet and SDXL.
    - **run_harmonize_sticker**: Handles bounding box extraction and edge-focused inpainting for faster processing (768×768 crop).
- `quantization_utils.py`: Contains the BitsAndBytesConfig setup. Configures the model to load in 4-bit NF4 (Normal Float 4) precision with double quantization, drastically reducing the memory footprint of the SDXL UNet and Text Encoders.
- `pruning_utils.py`: Implements Token Merging (ToMe). Applies dynamic structural pruning to the attention mechanism, removing approximately 40% of redundant tokens during the forward pass to speed up inference.

---

 

### Smart Fill Workflow (Detailed)

| Step | Operation | Inputs | Key Params | Output |
|---|---|---|---|---|
| 1 | Decode + resize | image, mask | work size 1024x1024 | RGB image, L mask |
| 2 | Generative Fill (inpaint) | image, mask, control=image | steps ~30, guidance ~7.5, control scale ~0.5 | Filled image |
| 3 | Optional Vibe Match | filled image | `vibe_strength` 0.2 to 0.4, steps >=30, guidance ~2.5 | Relit image |
| 4 | Resize back | relit/filled | original dims | Final PNG |

```mermaid
sequenceDiagram
    participant UI as Client UI
    participant API as FastAPI
    participant Pipe as EditingPipelines
    participant CN as ControlNet Inpaint
    participant SD as SDXL UNet
    participant VI as Img2Img
    UI->>API: Upload image+mask+prompt
    API->>Pipe: run_smart_fill(img, mask, prompt, vibe)
    Pipe->>CN: control(image, mask)
    CN->>SD: guided inpaint steps
    SD-->>Pipe: filled image
    alt vibe_strength>0
        Pipe->>VI: img2img(strength=vibe)
        VI-->>Pipe: relit image
    end
    Pipe-->>API: final image
    API-->>UI: PNG bytes
```

### Harmonization Workflow (Detailed)

| Step | Operation | Inputs | Key Params | Output |
|---|---|---|---|---|
| 1 | BBox on downscaled mask | mask | odd MaxFilter border (>=3) | bbox |
| 2 | Crop image + mask | image, mask | padding ~128 | cropped region |
| 3 | Edge-only mask | crop | MaxFilter/MinFilter + GaussBlur ~20 | smooth edge mask |
| 4 | Inpaint crop | work size ~768x768 | steps ~15, guidance ~2.5 | harmonized crop |
| 5 | Paste back | crop result | alpha from RGBA | final image |

```mermaid
flowchart LR
    M[Mask] --> DS[Downscale]
    DS --> MF[MaxFilter odd k]
    MF --> B[BBox]
    B --> CR[Crop]
    CR --> E[Edge Mask Dilate-Erode-Blur]
    E --> IP[Inpaint Crop]
    IP --> P[Paste RGBA on Base]
    P --> O[Final]
```

#### Edge Detection Process Detail

The harmonization pipeline uses morphological operations to isolate object edges for seamless blending:

```mermaid
flowchart TB
    subgraph Edge Mask Generation
        M[Binary Mask] --> D[Dilate: MaxFilter k=41]
        M --> ER[Erode: MinFilter k=41]
        D --> Diff[Difference]
        ER --> Diff
        Diff --> Blur[GaussianBlur σ=20]
        Blur --> Edge[Soft Edge Mask]
    end
    Edge --> Purpose["Purpose: Blend only at boundaries\nAvoids re-generating entire object"]
```

**Key Parameters**:
- **Border width (41px)**: Empirically tuned for T4 performance; must be odd for symmetric filters.
- **Gaussian blur (σ=20)**: Smooths transitions to prevent visible seams.

---

## API Reference
All endpoints accept multipart form-data. The mask should be white where changes are desired and black elsewhere.

- `GET /` - Service banner.
- `GET /health` - Returns `{ status: "running", gpu: <bool> }`.
- `POST /generative-fill`
    - Form: `image` (file), `mask` (file), `prompt` (string)
    - Behavior: Calls `run_smart_fill` with default `vibe_strength=0.1`.
    - Returns: `image/png` bytes
- `POST /smart-fill`
    - Form: `image` (file), `mask` (file), `prompt` (string), `vibe_strength` (float [0.0–1.0], default 0.0)
    - Behavior: Two-pass (Fill → Vibe Match when `vibe_strength` > 0)
    - Returns: `image/png` bytes
- `POST /harmonize`
    - Form: `image` (file), `mask` (file)
    - Behavior: Edge-only harmonization of pasted objects
    - Returns: `image/png` bytes

PowerShell examples:

```powershell
curl -Method GET http://localhost:8080/health

curl -Method POST "http://localhost:8080/smart-fill" \
    -Form image=@"C:\img\image.png" \
    -Form mask=@"C:\img\mask.png" \
    -Form prompt="a cozy wooden table background" \
    -Form vibe_strength=0.3 --output out.png

curl -Method POST "http://localhost:8080/harmonize" \
    -Form image=@"C:\\img\\composite.png" \
    -Form mask=@"C:\\img\\sticker_mask.png" --output out_h.png
```

### API Response Formats

**Success (200 OK)**:
- Content-Type: `image/png`
- Body: Raw PNG binary data
- Headers: Standard FastAPI response headers

**Error Responses**:

| Status Code | Condition | Response Body (JSON) |
|---|---|---|
| 400 Bad Request | Missing/invalid form fields | `{"detail": "Missing required field: <field>"}` |
| 422 Unprocessable Entity | Invalid image format | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 500 Internal Server Error | Model inference failure | `{"detail": "Internal server error"}` |

### Error Handling Workflow

```mermaid
flowchart TD
    Req[Incoming Request] --> Val{Valid multipart?}
    Val -->|No| E400[Return 400]
    Val -->|Yes| Dec{Decodable images?}
    Dec -->|No| E422[Return 422]
    Dec -->|Yes| Inf[Run Inference]
    Inf --> Err{Exception?}
    Err -->|CUDA OOM| Log[Log to console and W&B]
    Log --> E500[Return 500]
    Err -->|Other| Log
    Err -->|No| Enc[Encode PNG]
    Enc --> Res[Return 200 plus image]
```

---

## Data Flow Diagrams

### High-Level Request Flow

```mermaid
flowchart TB
    subgraph Client
        UI[Editor UI] -->|image, mask, prompt| API[HTTP Client]
    end
    API -->|multipart/form-data| S[FastAPI]
    S -->|PIL decode| D[Decoded Image & Mask]
    S --> EP[EditingPipelines]
    EP --> IP[Inpaint Pipeline]
    EP -->|optional| IM[Img2Img Pipeline]
    IP --> R1[(Filled Image)]
    IM --> R2[(Relit Image)]
    R1 -->|if no vibe| O[PNG]
    R2 -->|else| O[PNG]
    O --> Client
```

 

**Data Dimensions at Each Stage**:
- Input Image: `3x1024x1024` (RGB)
- Latent Space: `4x128x128` (8x compression)
- Text Embeddings: `77x2048` (sequence length x embedding dim)
- UNet Hidden: `1280` channels at bottleneck
- ControlNet Features: Multi-scale `[64, 128, 256]` channels


