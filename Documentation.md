# Image Editing with Stable Diffusion XL Featuring Generative Fill and Edge Harmonization

---

## Overview

**Features:**
- Text-guided Generative Fill
- Edge-aware Harmonization
- Two-pass Smart Fill with lighting/style matching
- Low VRAM optimized

**Tech Stack:**
- Stable Diffusion XL
- FastAPI REST API

## Architecture

**Core Files:**
- `server.py` - API server
- `editing_pipelines_fill.py` - Image editing pipeline
- `quantization_utils.py` - Model optimization
- `pruning_utils.py` - Performance optimization
- `requirements.txt` - Dependencies

![SDXL Architecture](assets/SDXL_Architecture.jpg)

```mermaid
flowchart LR
    Client --> FastAPI
    FastAPI --> EditingPipeline
    EditingPipeline --> SDXL_Inpaint[SDXL Inpaint]
    EditingPipeline --> SDXL_Img2Img[SDXL Img2Img]
    SDXL_Inpaint --> Result
    SDXL_Img2Img --> Result
    Result --> Response[PNG Response]
```

---

## Parameters

- `prompt` - Desired content description
- `mask` - White for edit region, black for preserve
- `vibe_strength` - Style matching intensity (0.0-1.0)

**Recommended Values:**

| Task | Steps | Vibe Strength |
|------|-------|---------------|
| Replace background | 30 | 0.3 |
| Add object | 30 | 0.2 |
| Fast relight | 20 | 0.1 |
| Harmonize edges | 15 | 0 |

---

## Performance

- **VRAM:** 8-9 GB
- **Speed:** First run downloads models, subsequent runs are faster
- **Optimization:** Adjust steps and vibe_strength parameters for quality/speed tradeoff

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

```powershell
python server.py
```

Server: `http://localhost:8080`  
API Docs: `http://localhost:8080/docs`

**Requirements:** GPU recommended, models auto-download on first run.

---

## Workflow

**Smart Fill:**
1. Upload image, mask, and prompt
2. Generate content in masked area
3. Apply optional style matching
4. Return processed image

**Harmonization:**
1. Detect object boundaries
2. Blend edges seamlessly

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Pipeline
    participant Inpaint
    participant Img2Img

    Client->>FastAPI: Upload image, mask, prompt
    FastAPI->>Pipeline: Process request
    Pipeline->>Inpaint: Generate fill
    Inpaint-->>Pipeline: Filled image
    alt vibe_strength > 0
        Pipeline->>Img2Img: Apply vibe match
        Img2Img-->>Pipeline: Adjusted image
    end
    Pipeline-->>FastAPI: Final result
    FastAPI-->>Client: Return PNG
```

---

## API Endpoints

**`GET /health`**  
Health check

**`POST /generative-fill`**  
Params: `image`, `mask`, `prompt`  
Returns: PNG image

**`POST /smart-fill`**  
Params: `image`, `mask`, `prompt`, `vibe_strength`  
Returns: PNG image

**`POST /harmonize`**  
Params: `image`, `mask`  
Returns: PNG image

### Examples

```powershell
curl -Method GET http://localhost:8080/health

curl -Method POST "http://localhost:8080/smart-fill" \
	-Form image=@"C:\img\image.png" \
	-Form mask=@"C:\img\mask.png" \
	-Form prompt="a cozy wooden table background" \
	-Form vibe_strength=0.3 --output out.png

curl -Method POST "http://localhost:8080/harmonize" \
	-Form image=@"C:\img\composite.png" \
	-Form mask=@"C:\img\sticker_mask.png" --output out_h.png
```

---

## Performance Charts

### GPU Utilization
![GPU Utilization](assets/Section-4-Panel-19-zi0m568sl.png)
![Power Usage](assets/Section-4-Panel-20-uy8a7yk1w.png)
![Temperature](assets/Section-4-Panel-21-0w4dex6yu.png)

### System Metrics
![Latency](assets/Section-2-Panel-0-qfho7iza6.png)
![Steps](assets/Section-2-Panel-1-w0lfq6r37.png)
![VRAM](assets/Section-2-Panel-2-nnkqs6j8d.png)
![CPU Usage](assets/Section-2-Panel-3-xxtm21cpo.png)
![RAM Usage](assets/Section-2-Panel-4-byl7ls1rf.png)

### Detailed Metrics
![TFLOPS](assets/Section-2-Panel-5-t42wqk4og.png)
![Throughput](assets/Section-2-Panel-6-lquc9hls9.png)
![Disk I/O](assets/Section-2-Panel-7-xyr9djojz.png)
![Network](assets/Section-2-Panel-8-nm77dfanm.png)
![Memory Allocated](assets/Section-2-Panel-9-h4cl1t1tn.png)

### Additional Analysis
![GPU Power](assets/Section-4-Panel-0-t9w5eqvok.png)
![Temperature Detail](assets/Section-4-Panel-1-8aucfv5zf.png)
![Clock Speed](assets/Section-4-Panel-2-fif6xyq1o.png)
![Utilization Detail](assets/Section-4-Panel-3-9kcyle0jf.png)
![Memory Usage](assets/Section-4-Panel-4-9hbgsom47.png)
![Memory Access](assets/Section-4-Panel-5-6fu4y3q94.png)
![Bandwidth](assets/Section-4-Panel-6-0usjrj1gh.png)
![Compute](assets/Section-4-Panel-7-h47es1qad.png)
![System RAM](assets/Section-4-Panel-8-qat4uz36p.png)
![CPU Load](assets/Section-4-Panel-9-ivlkr7616.png)
![Disk Read](assets/Section-4-Panel-10-buu7ktx3j.png)
![Disk Write](assets/Section-4-Panel-11-0v2rimcx0.png)
![Network Sent](assets/Section-4-Panel-12-2zhn685ri.png)
![Network Received](assets/Section-4-Panel-13-eczzi41t8.png)
![Process Memory](assets/Section-4-Panel-14-6x1e4g9na.png)
![Thread Count](assets/Section-4-Panel-15-n4oqhxe9w.png)
![Latency Distribution](assets/Section-4-Panel-16-xjq0oyce6.png)
![Steps Distribution](assets/Section-4-Panel-17-wjfeuo292.png)
![TFLOPS Distribution](assets/Section-4-Panel-18-oldxwnlb3.png)


