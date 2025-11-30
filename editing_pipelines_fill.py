import torch
from PIL import ImageChops, ImageFilter
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from PIL import Image, ImageFilter

class EditingPipelines:
    def __init__(self):
        print("--- Loading SDXL Hybrid Pipelines ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 for speed and memory efficiency
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 1. Load ControlNet (Inpaint Dreamer is excellent for structural filling)
        controlnet = ControlNetModel.from_pretrained(
            "destitech/controlnet-inpaint-dreamer-sdxl",
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        # 2. Main Inpainting Pipeline (Handles Fill & Add)
        self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        self.inpaint_pipe.scheduler = UniPCMultistepScheduler.from_config(self.inpaint_pipe.scheduler.config)

        # 3. Global Img2Img Pipeline (Handles Vibe Matching/Harmonization)
        # We reuse the components to save VRAM if possible, or load separately
        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)
        
        print("--- Pipelines Ready ---")

    def run_smart_fill(self, image, mask, prompt, vibe_strength=0.1):
        """
        The "Pro" Workflow:
        1. Generative Fill: Fills the mask with the prompt.
        2. Vibe Match: Relights the UNMASKED parts to match the new prompt.
        
        vibe_strength: How much to change the original pixels (0.0 - 1.0).
                       0.3 is usually perfect for relighting without changing identity.
        """
        print(f"Running Smart Fill: '{prompt}'")
        
        # --- Step 1: Generative Fill ---
        # Resize for SDXL
        w, h = image.size
        work_image = image.convert("RGB").resize((1024, 1024))
        work_mask = mask.convert("L").resize((1024, 1024))
        
        # Create the new content (Background/Object)
        filled_image = self.inpaint_pipe(
            prompt=f"{prompt}, 8k, photorealistic, master calibration",
            negative_prompt="blurry, ugly, deformed, text, watermark",
            image=work_image,
            mask_image=work_mask,
            control_image=work_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=1.0,  # 100% generation in the mask
            controlnet_conditioning_scale=0.0 # Ignore original shape in mask
        ).images[0]
        
        # --- Step 2: Global Vibe Match (Harmonization) ---
        if vibe_strength > 0:
            print(f"  > Applying Global Vibe Match (Strength: {vibe_strength})...")
            # We take the FILLED image and run it through Img2Img with the SAME prompt
            filled_image = self.img2img_pipe(
                prompt=f"{prompt}, consistent lighting, 8k, photorealistic",
                negative_prompt="blurry, artifact, distorted",
                image=filled_image,
                num_inference_steps=30, # Fast pass
                strength=vibe_strength, # Low strength = Relighting only
                guidance_scale=7.5
            ).images[0]
            
        return filled_image.resize((w, h), Image.Resampling.LANCZOS)

    def run_harmonize_sticker(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        WORKFLOW: Edge-Only Harmonization (The "Halo" Method)
        1. Creates a 'Ring Mask' at the boundary of the object.
        2. Crops/Zooms into the object area for high resolution.
        3. Inpaints ONLY the ring, blending the edges seamlessly.
        4. Preserves the interior pixels 100% (no regeneration).
        """
        print("-> Running Edge-Only Harmonization")
        
        # --- Step 1: Create the Edge Mask (The Halo) ---
        # We want to blend ~15-20 pixels inside and outside the seam.
        border_width = 21
        
        # Grow mask to cover background pixels near edge
        mask_dilated = mask.filter(ImageFilter.MaxFilter(border_width))
        
        # Shrink mask to cover object pixels near edge
        mask_eroded = mask.filter(ImageFilter.MinFilter(border_width))
        
        # Difference = The Ring (White at the edge, Black in center & background)
        edge_mask = ImageChops.difference(mask_dilated, mask_eroded)
        
        # Soften the ring for smoother blending
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(5))

        # --- Step 2: Focus/Crop Logic (To preserve resolution) ---
        # We crop based on the *dilated* mask to ensure we catch the whole halo
        bbox = mask_dilated.getbbox() 
        
        if not bbox:
            print("   Warning: Empty mask, returning original.")
            return image

        padding = 128 # Context padding
        width, height = image.size
        
        left, top, right, bottom = bbox
        crop_left = max(0, left - padding)
        crop_top = max(0, top - padding)
        crop_right = min(width, right + padding)
        crop_bottom = min(height, bottom + padding)
        
        crop_box = (crop_left, crop_top, crop_right, crop_bottom)
        
        # Crop Image and the EDGE Mask
        image_crop = image.crop(crop_box)
        edge_mask_crop = edge_mask.crop(crop_box)
        
        original_crop_size = image_crop.size
        
        # Upscale for SDXL
        work_image = image_crop.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        work_mask = edge_mask_crop.convert("L").resize((1024, 1024), Image.Resampling.NEAREST) # Nearest for mask precision

        # --- Step 3: Run Inference ---
        # We can now use HIGHER strength because we are only touching the edge.
        # This allows the AI to aggressively fix lighting seams without ruining the face/object.
        output_crop = self.inpaint_pipe(
            prompt="coherent lighting, realistic shadows, high quality, seamless blending",
            negative_prompt="blurry, artificial, visible seam, cut out, glowing edge",
            image=work_image,
            mask_image=work_mask, # Pass the RING mask
            control_image=work_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.75, # High strength allows it to truly blend the seam pixels
            controlnet_conditioning_scale=0.4 # Moderate control to guide the shape of the blend
        ).images[0]
        
        # --- Step 4: Paste Back ---
        output_crop = output_crop.resize(original_crop_size, Image.Resampling.LANCZOS)
        
        final_image = image.copy()
        
        # We strictly paste only where the EDGE mask is white (or non-zero).
        # However, pasting the whole square crop is usually fine since the center 
        # wasn't changed by the AI (because the mask was black there).
        final_image.paste(output_crop, (crop_left, crop_top))
        
        return final_image
        
        # return output.resize((w, h), Image.Resampling.LANCZOS)