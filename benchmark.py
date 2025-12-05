import torch
import os
from datasets import load_dataset
from diffusers import (
    StableDiffusionXLInstructPix2PixPipeline,
    AutoPipelineForInpainting,
    EulerDiscreteScheduler
)
from editing_pipelines_fill import EditingPipelines  # Your Pipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
RESULTS_DIR = "benchmark_results"
DEVICE = "cuda"

# Define the models to benchmark
MODELS = {
    # 1. YOUR MODEL
    "kortex_ours": {
        "type": "kortex",
        "id": "local"
    },
    # 2. INSTRUCTION MODEL (The true competitor)
    "sdxl_instructpix2pix": {
        "type": "instruct",
        "id": "diffusers/sdxl-instructpix2pix-768"
    },
    # 3. STANDARD SDXL FINE-TUNES (Loaded as Inpainters)
    "realvis_v4": {
        "type": "inpaint_standard",
        "id": "SG161222/RealVisXL_V4.0"
    },
    "juggernaut_xl": {
        "type": "inpaint_standard",
        "id": "RunDiffusion/Juggernaut-XL-v9"
    },
    "thinkdiffusion_xl": {
        "type": "inpaint_standard",
        "id": "ThinkDiffusion/ThinkDiffusionXL"
    },
    # Note: "Realistic Vision" is usually SD1.5. 
    # We use RealVisXL which is the SDXL equivalent by the same creator.
}

def load_pipeline(model_key, model_config):
    """Loads the appropriate pipeline based on type."""
    print(f"\n--- Loading {model_key} ---")
    
    if model_config["type"] == "kortex":
        return EditingPipelines() # Your class handles loading
        
    elif model_config["type"] == "instruct":
        pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            model_config["id"], 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )
    
    elif model_config["type"] == "inpaint_standard":
        # We wrap standard SDXL checkpoints in an Inpainting Pipeline
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_config["id"], 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
        
    # Common Optimizations
    if model_config["type"] != "kortex":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(DEVICE)
        # pipe.enable_model_cpu_offload() # Uncomment if VRAM is tight ( < 24GB )
    
    return pipe

def run_benchmark():
    # 1. Setup Data
    dataset = load_dataset("osunlp/MagicBrush", split="dev[:35]") # Use 'dev' for speed
    
    # 2. Iterate through ALL models
    for model_name, config in MODELS.items():
        
        # Output setup
        out_path = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(out_path, exist_ok=True)
        
        # Init FID for this model run
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
        
        # Load Model
        try:
            pipe = load_pipeline(model_name, config)
        except Exception as e:
            print(f"Skipping {model_name} due to load error: {e}")
            continue

        print(f"Running Inference for {model_name}...")
        
        for i, item in tqdm(enumerate(dataset), total=len(dataset)):
            # Prepare Inputs
            source = item['source_img'].convert("RGB").resize((1024, 1024))
            mask = item['mask_img'].convert("L").resize((1024, 1024))
            target = item['target_img'].convert("RGB").resize((1024, 1024))
            prompt = item['instruction'] # "Make the cat red"

            # --- INFERENCE ---
            generated = None
            with torch.no_grad():
                if config["type"] == "kortex":
                    # Your Pipeline
                    generated = pipe.run_smart_fill(source, mask, prompt, vibe_strength=0.0, use_refiner=False)
                
                elif config["type"] == "instruct":
                    # IP2P: Uses Image + Prompt (No Mask natively, but we can paste back)
                    # IP2P edits the WHOLE image. To be fair to Inpainting models, 
                    # we should probably composite it back using the mask.
                    res = pipe(
                        prompt=prompt, 
                        image=source, 
                        num_inference_steps=30, 
                        image_guidance_scale=1.5
                    ).images[0]
                    
                    # Optional: Composite back to be fair to mask-based models
                    # generated = Image.composite(res, source, mask) 
                    generated = res # Or keep raw to test instruction following
                    
                elif config["type"] == "inpaint_standard":
                    # Fine-tunes: Use Image + Mask + Prompt
                    generated = pipe(
                        prompt=prompt,
                        image=source,
                        mask_image=mask,
                        num_inference_steps=30,
                        strength=0.99 # High strength needed for significant changes
                    ).images[0]

            # --- FID ACCUMULATION ---
            generated = generated.resize((1024, 1024))
            
            # Save sample (optional)
            if i < 5: # Save first 5 for visual check
                generated.save(f"{out_path}/sample_{i}.png")

            # Convert to tensors
            real_t = pil_to_tensor(target).unsqueeze(0).to(DEVICE) / 255.0
            fake_t = pil_to_tensor(generated).unsqueeze(0).to(DEVICE) / 255.0
            
            fid.update(real_t, real=True)
            fid.update(fake_t, real=False)
            
            # Clean up VRAM per image if needed
            # torch.cuda.empty_cache()

        # Compute & Print
        score = fid.compute()
        print(f"\n>> FINAL FID SCORE for {model_name}: {score.item():.4f}")
        
        # Cleanup Model to free VRAM for next one
        del pipe
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_benchmark()