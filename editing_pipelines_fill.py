import torch
import time
import threading
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

# --- LOGGING & MONITORING IMPORTS ---
try:
    import wandb
except ImportError:
    wandb = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

# Model IDs
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_MODEL = "destitech/controlnet-inpaint-dreamer-sdxl"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"

# --- BACKGROUND MONITOR ---
class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.ram_samples = []
        self.cpu_samples = []
        self.ram_start_gb = 0.0

    def start(self):
        if psutil:
            self.ram_start_gb = psutil.virtual_memory().used / (1024**3)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _monitor(self):
        while not self.stop_event.is_set():
            if psutil:
                self.ram_samples.append(psutil.virtual_memory().used / (1024**3))
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def get_stats(self):
        if not self.ram_samples:
            return self.ram_start_gb, self.ram_start_gb, self.ram_start_gb, 0.0
        
        ram_peak = max(self.ram_samples)
        ram_avg = sum(self.ram_samples) / len(self.ram_samples)
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        
        return self.ram_start_gb, ram_peak, ram_avg, cpu_avg

class EditingPipelines:
    def __init__(self):
        print("--- Loading Optimized SDXL Pipeline (Quality Focused) ---")
        
        # 0. Initialize WandB & NVML
        if wandb and wandb.run is None:
            print("... Initializing WandB")
            wandb.init(project="sdxl-inference-monitor", name="server-instance", resume="allow")
            
        self.pynvml_handle = None
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: Could not init pynvml for GPU utilization: {e}")

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
        vae.enable_slicing() # fits in the VRAM

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
        apply_token_pruning(self.inpaint_pipe, ratio=0.15)

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

        print("--- Optimization Complete: 4-bit | 15% Pruned | 10 Steps ---")

    def _get_gpu_util(self):
        """Helper to get GPU Utilization from pynvml"""
        if self.pynvml_handle:
            try:
                return pynvml.nvmlDeviceGetUtilizationRates(self.pynvml_handle).gpu
            except:
                return 0
        return 0

    def _log_metrics(self, start_time, steps, step_name="Inference"):
        """
        Kept for console output.
        Real logging is now handled by WandB logic in individual methods.
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
            latency = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            estimated_ops_per_step = 5.0 
            total_ops = estimated_ops_per_step * steps
            throughput_tflops = total_ops / latency
            estimated_tops = throughput_tflops * 2.0 
            
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
        
        # Start Background Monitor
        monitor = ResourceMonitor()
        monitor.start()
        t0 = time.time()
        
        total_steps_executed = 0
        filled_image = None
        
        try:
            work_image = image.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
            work_mask = mask.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
            
            # --- Step 1: Generative Fill ---
            print("  > Starting Generative Fill (30 Steps)...")
            if self.device == "cuda": torch.cuda.reset_peak_memory_stats()
            
            steps_1 = 30
            filled_image = self.inpaint_pipe(
                prompt=f"{prompt}, 8k, photorealistic, master calibration",
                negative_prompt="blurry, ugly, deformed, text, watermark, low quality",
                image=work_image,
                mask_image=work_mask,
                control_image=work_image,
                num_inference_steps=steps_1,  
                guidance_scale=7.5,
                strength=1.0,
                controlnet_conditioning_scale=0.5
            ).images[0]
            
            total_steps_executed += steps_1
            self._log_metrics(t0, steps=steps_1, step_name="Generative Fill")
            
            # --- Step 2: Vibe Match ---
            if vibe_strength > 0:
                print(f"  > Starting Vibe Match (Strength: {vibe_strength})...")
                safe_steps = max(30, int(1.0 / vibe_strength) + 1)
                
                # We calculate approx steps that run based on strength
                # img2img runs for (strength * num_inference_steps)
                actual_steps_2 = int(safe_steps * vibe_strength)
                
                filled_image = self.img2img_pipe(
                    prompt=f"{prompt}, consistent lighting, 8k, photorealistic",
                    negative_prompt="blurry, artifact, distorted",
                    image=filled_image,
                    num_inference_steps=safe_steps, 
                    strength=vibe_strength,
                    guidance_scale=2.5
                ).images[0]
                
                total_steps_executed += actual_steps_2
                self._log_metrics(time.time(), steps=actual_steps_2, step_name="Vibe Match")

        finally:
            # Stop Monitor & Collect Metrics
            monitor.stop()
            
            # 1. Resource Metrics
            ram_start, ram_peak, ram_avg, cpu_avg = monitor.get_stats()
            latency = time.time() - t0
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            gpu_util = self._get_gpu_util()
            
            # 2. Compute Estimation (TFLOPS)
            # Formula: 2 * Params * Steps * ResolutionFactor
            MODEL_PARAMS = 2.6e9  # Placeholder: Approx SDXL UNet params
            res_factor = (1024 * 1024) / (1024 * 1024) # Processing done at 1024px
            
            estimated_tflops = (2 * MODEL_PARAMS * total_steps_executed * res_factor) / 1e12

            # 3. Log to WandB
            if wandb:
                wandb.log({
                    "request_type": "smart_fill",
                    "prompt": prompt,
                    "latency_s": latency,
                    "total_steps": total_steps_executed,
                    
                    # Resources
                    "ram_start_gb": ram_start,
                    "ram_peak_gb": ram_peak,
                    "ram_avg_gb": ram_avg,
                    "cpu_usage_avg_percent": cpu_avg,
                    "gpu_mem_peak_gb": gpu_mem,
                    "gpu_util_percent": gpu_util,
                    
                    # Compute
                    "estimated_tflops_total": estimated_tflops,
                    "estimated_tflops_per_sec": estimated_tflops / latency if latency > 0 else 0
                })

        return filled_image.resize((w, h), Image.Resampling.LANCZOS)

    def run_harmonize_sticker(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        print("-> Running Edge-Only Harmonization (Optimized Pre-processing)")
        
        # 1. Start Background Monitor
        monitor = ResourceMonitor()
        monitor.start()
        t0 = time.time()
        
        # Optimization Parameters
        process_size = (768, 768)
        total_steps = 15
        
        # CLEANUP: Force garbage collection before heavy work to prevent "subsequent" slowdowns
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_image = image # Default return if bbox fails
        
        # Hardcoded prompt used for logging consistency
        prompt_used = "seamless blending, smooth edges, anti-aliased, coherent lighting, realistic shadows, high quality"

        try:
            width, height = image.size
            border_width = 41
            
            # BBox calculation
            analysis_scale = 1024 / max(width, height)
            
            if analysis_scale < 1.0:
                # Resize mask for analysis
                w_small = int(width * analysis_scale)
                h_small = int(height * analysis_scale)
                mask_small = mask.resize((w_small, h_small), Image.Resampling.NEAREST)
                
                # Calculate scaled border
                scaled_border = int(border_width * analysis_scale)
                
                # Ensure it is at least 3 and always ODD
                if scaled_border < 3:
                    border_small = 3
                elif scaled_border % 2 == 0:
                    border_small = scaled_border + 1 # Turn 10 into 11
                else:
                    border_small = scaled_border
                
                # Fast Filter on small mask
                mask_dilated_small = mask_small.filter(ImageFilter.MaxFilter(border_small))
                bbox_small = mask_dilated_small.getbbox()
                
                if not bbox_small:
                    print("No bbox found in mask")
                    return image

                # Scale BBox back up
                bbox = (
                    int(bbox_small[0] / analysis_scale),
                    int(bbox_small[1] / analysis_scale),
                    int(bbox_small[2] / analysis_scale),
                    int(bbox_small[3] / analysis_scale)
                )
            else:
                # Image is already small, run normally
                # Ensure original border_width is odd (41 is odd, so this is safe)
                safe_border = border_width if border_width % 2 != 0 else border_width + 1
                mask_dilated = mask.filter(ImageFilter.MaxFilter(safe_border))
                bbox = mask_dilated.getbbox()
                if not bbox: return image

        
            padding = 128
            crop_box = (
                max(0, bbox[0]-padding), 
                max(0, bbox[1]-padding), 
                min(width, bbox[2]+padding), 
                min(height, bbox[3]+padding)
            )
            
            # Now we crop the ORIGINAL high-res image and mask
            image_crop = image.crop(crop_box)
            mask_crop = mask.crop(crop_box)
            
            # Now we perform the expensive filters ONLY on the cropped area (much smaller)
            # We must re-calculate dilated/eroded on the crop to get the edge mask
            mask_dilated_crop = mask_crop.filter(ImageFilter.MaxFilter(border_width))
            mask_eroded_crop = mask_crop.filter(ImageFilter.MinFilter(border_width))
            
            # Smooth alpha transitions
            edge_mask = ImageChops.difference(mask_dilated_crop, mask_eroded_crop).filter(ImageFilter.GaussianBlur(20))
            
            # Resize for Inference
            work_image = image_crop.convert("RGB").resize(process_size, Image.Resampling.LANCZOS)
            work_mask = edge_mask.convert("L").resize(process_size, Image.Resampling.NEAREST)
            original_crop_size = image_crop.size

            # --- Run Inference ---
            if self.device == "cuda": torch.cuda.reset_peak_memory_stats()

            output_crop = self.inpaint_pipe(
                prompt=prompt_used,
                negative_prompt="jagged, zig-zag, pixelated, blurry, artificial, visible seam, cut out, glowing edge",
                image=work_image,
                mask_image=work_mask,
                control_image=work_image,
                num_inference_steps=total_steps,
                guidance_scale=2.5,
                strength=0.75, 
                controlnet_conditioning_scale=0.4
            ).images[0]
            
            # Console logging
            self._log_metrics(t0, steps=total_steps, step_name="Harmonization")
            
            output_crop = output_crop.resize(original_crop_size, Image.Resampling.LANCZOS)
            final_image = image.copy()
            final_image.paste(output_crop, (crop_box[0], crop_box[1]), output_crop.convert("RGBA"))
            
        finally:
            monitor.stop()
            
            # Logging logic remains the same...
            ram_start, ram_peak, ram_avg, cpu_avg = monitor.get_stats()
            latency = time.time() - t0
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            gpu_util = self._get_gpu_util()
            
            MODEL_PARAMS = 2.6e9
            res_factor = (process_size[0] * process_size[1]) / (1024 * 1024) 
            estimated_tflops = (2 * MODEL_PARAMS * total_steps * res_factor) / 1e12
            
            if wandb:
                wandb.log({
                    "request_type": "harmonize",
                    "prompt": prompt_used,
                    "latency_s": latency,
                    "total_steps": total_steps,
                    
                    # Resources
                    "ram_start_gb": ram_start,
                    "ram_peak_gb": ram_peak,
                    "ram_avg_gb": ram_avg,
                    "cpu_usage_avg_percent": cpu_avg,
                    "gpu_mem_peak_gb": gpu_mem,
                    "gpu_util_percent": gpu_util,
                    
                    # Compute
                    "estimated_tflops_total": estimated_tflops,
                    "estimated_tflops_per_sec": estimated_tflops / latency if latency > 0 else 0,
                    
                    # Extra info specific to harmonization
                    "resolution": f"{process_size[0]}x{process_size[1]}"
                })
        
        return final_image
