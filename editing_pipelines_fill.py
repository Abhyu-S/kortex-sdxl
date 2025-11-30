import torch
import time
from PIL import Image, ImageChops, ImageFilter
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
# Import our custom modules
from quantization_utils import get_quantization_config
from pruning_utils import apply_token_pruning

# Model IDs
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL = "destitech/controlnet-inpaint-dreamer-sdxl"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix" 

class EditingPipelines:
    def __init__(self):
        print("--- Loading Optimized SDXL Pipeline (Quality Focused) ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. QUANTIZATION: Get the NF4 Config
        quant_config = get_quantization_config()

        # 2. Load ControlNet (Float16)
        print("... Loading ControlNet (FP16)")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        controlnet.to(self.device)

        # --- MANUAL LOADING WITH QUANTIZATION ---
        print("... Loading SDXL Components with 4-bit (NF4)")

        # Load UNet with Quantization
        unet = UNet2DConditionModel.from_pretrained(
            BASE_MODEL,
            subfolder="unet",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Load Text Encoders with Quantization
        text_encoder_1 = CLIPTextModel.from_pretrained(
            BASE_MODEL,
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            BASE_MODEL,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Load VAE (FP16)
        vae = AutoencoderKL.from_pretrained(
            VAE_MODEL,
            torch_dtype=torch.float16
        )
        vae.to(self.device)
        vae.enable_slicing() 

        # Load Tokenizers
        print("... Loading Tokenizers")
        tokenizer_1 = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            subfolder="tokenizer",
            use_fast=False
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            subfolder="tokenizer_2",
            use_fast=False
        )
        # --- END MANUAL LOADING ---

        # 3. Assemble Main Pipeline
        print("... Assembling StableDiffusionXLControlNetInpaintPipeline")
        self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            controlnet=controlnet,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=EulerDiscreteScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler"),
        )
        
        self.inpaint_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.inpaint_pipe.scheduler.config, 
            timestep_spacing="trailing"
        )

        # 4. PRUNING: Reduced ratio for better quality
        # 0.2 means removing 20% of tokens (vs 40% before). 
        # This keeps more texture details while still giving a small speedup.
        apply_token_pruning(self.inpaint_pipe, ratio=0.4)

        # 5. Assemble Img2Img Pipeline
        print("... Assembling Img2Img Pipeline")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet=unet,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=EulerDiscreteScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler"),
        )
        self.img2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.img2img_pipe.scheduler.config, 
            timestep_spacing="trailing"
        )
        # Pruning is already applied to the shared UNet

        print("--- Optimization Complete: 4-bit | 15% Pruned | 10 Steps ---")

    def _log_metrics(self, start_time, steps, step_name="Inference"):
        """
        Calculates and prints compute metrics:
        - Latency
        - Peak VRAM
        - Estimated Throughput (TFLOPs)
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
            
            # Time & Memory
            latency = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            
            # Compute Estimation (Ballpark for Feasibility Report)
            # SDXL 1024px is roughly 5 TFLOPs per step (FP16).
            # We estimate the 'Effective TFLOPs' delivered by the GPU.
            estimated_ops_per_step = 5.0 # Trillions of ops
            total_ops = estimated_ops_per_step * steps
            throughput_tflops = total_ops / latency
            
            # TOPS (Trillions of Operations Per Second)
            # For INT8/INT4, TOPS is often 2x-4x TFLOPs depending on hardware instructions.
            # We estimate TOPS as a function of the mixed-precision throughput.
            estimated_tops = throughput_tflops * 2.0 # Conservative estimate for quantized ops
            
            print(f"[{step_name}] Metrics:")
            print(f"  > Latency:   {latency:.4f} s")
            print(f"  > Peak VRAM: {max_mem:.2f} GB")
            print(f"  > Est. Perf: {throughput_tflops:.2f} TFLOPs | {estimated_tops:.2f} TOPS")
        else:
            latency = time.time() - start_time
            print(f"[{step_name}] Latency: {latency:.4f}s (CPU)")

    def run_smart_fill(self, image, mask, prompt, vibe_strength=0.1):
        print(f"Running Smart Fill: '{prompt}'")
        w, h = image.size
        
        work_image = image.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        work_mask = mask.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
        
        # --- Step 1: Generative Fill (10 Steps) ---
        print("  > Starting Generative Fill (10 Steps)...")
        if self.device == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        
        filled_image = self.inpaint_pipe(
            prompt=f"{prompt}, 8k, photorealistic, master calibration",
            negative_prompt="blurry, ugly, deformed, text, watermark, low quality",
            image=work_image,
            mask_image=work_mask,
            control_image=work_image,
            num_inference_steps=30,  
            guidance_scale=7.5,
            strength=1.0,
            controlnet_conditioning_scale=0.5
        ).images[0]
        
        self._log_metrics(t0, steps=30, step_name="Generative Fill")
        
        # --- Step 2: Vibe Match ---
        if vibe_strength < 0:
            print(f"  > Starting Vibe Match (Strength: {vibe_strength})...")
            if self.device == "cuda": torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            
            # Ensure at least 10 steps so low strength doesn't result in 0 steps
            safe_steps = max(30, int(1.0 / vibe_strength) + 1)
            
            filled_image = self.img2img_pipe(
                prompt=f"{prompt}, consistent lighting, 8k, photorealistic",
                negative_prompt="blurry, artifact, distorted",
                image=filled_image,
                num_inference_steps=safe_steps, 
                strength=vibe_strength,
                guidance_scale=2.5
            ).images[0]
            
            # Calculate actual steps executed (approximate)
            actual_steps = int(safe_steps * vibe_strength)
            self._log_metrics(t0, steps=actual_steps, step_name="Vibe Match")
            
        return filled_image.resize((w, h), Image.Resampling.LANCZOS)

    def run_harmonize_sticker(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("-> Running Edge-Only Harmonization")
        
        border_width = 41
        mask_dilated = mask.filter(ImageFilter.MaxFilter(border_width))
        bbox = mask_dilated.getbbox() 
        if not bbox: return image
        
        padding = 128
        width, height = image.size
        crop_box = (max(0, bbox[0]-padding), max(0, bbox[1]-padding), min(width, bbox[2]+padding), min(height, bbox[3]+padding))
        
        image_crop = image.crop(crop_box)
        mask_eroded = mask.filter(ImageFilter.MinFilter(border_width))
        edge_mask = ImageChops.difference(mask_dilated, mask_eroded).filter(ImageFilter.GaussianBlur(5)).crop(crop_box)
        
        work_image = image_crop.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        work_mask = edge_mask.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
        original_crop_size = image_crop.size

        # --- Run Inference ---
        if self.device == "cuda": torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        output_crop = self.inpaint_pipe(
            prompt="coherent lighting, realistic shadows, high quality, seamless blending",
            negative_prompt="blurry, artificial, visible seam, cut out, glowing edge",
            image=work_image,
            mask_image=work_mask,
            control_image=work_image,
            num_inference_steps=30,
            guidance_scale=2.5,
            strength=0.75, 
            controlnet_conditioning_scale=0.4
        ).images[0]
        
        self._log_metrics(t0, steps=30, step_name="Harmonization") # 10 * 0.75 ≈ 8 steps
        
        output_crop = output_crop.resize(original_crop_size, Image.Resampling.LANCZOS)
        final_image = image.copy()
        final_image.paste(output_crop, (crop_box[0], crop_box[1]), output_crop.convert("RGBA"))
        
        return final_image