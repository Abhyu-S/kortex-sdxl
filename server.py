import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from PIL import Image
import io
import torch

# Import our pipeline
from editing_pipelines_fill import EditingPipelines

# Initialize App & Model
app = FastAPI()
print("Initializing AI Server...")
editor = EditingPipelines()

def decode_image(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# --- ADDED: Root Endpoint to fix 404 Error ---
@app.get("/")
def read_root():
    return {"message": "GenAI Server is Running! Go to /docs for API UI."}

@app.get("/health")
def health_check():
    return {"status": "running", "gpu": torch.cuda.is_available()}

@app.post("/generative-fill")
async def generative_fill(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Endpoint for: Add Object, Replace Background
    """
    print(f"Received Fill Request: {prompt}")
    
    # 1. Decode Inputs
    pil_image = decode_image(await image.read())
    pil_mask = decode_image(await mask.read())
    
    # 2. Run Inference
    result = editor.run_smart_fill(pil_image, pil_mask, prompt)
    
    # 3. Return Image
    return Response(content=image_to_bytes(result), media_type="image/png")

@app.post("/harmonize")
async def harmonize(
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    """
    Endpoint for: Move Object (Lighting fix)
    Expects: 'image' to be the Composite (Background + Moved Sticker)
    Expects: 'mask' to be White ONLY where the sticker was moved to.
    """
    print(f"Received Harmonize Request")
    
    # 1. Decode Inputs
    pil_image = decode_image(await image.read())
    pil_mask = decode_image(await mask.read())
    
    # 2. Run Inference
    result = editor.run_harmonize_sticker(pil_image, pil_mask)
    
    # 3. Return Image
    return Response(content=image_to_bytes(result), media_type="image/png")

@app.post("/smart-fill")
async def smart_fill(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    vibe_strength: float = Form(0.0) # Default 30% relighting
):
    """
    Use this for: Add Object, Replace Background.
    It performs the 2-pass 'Fill-and-Blend' automatically.
    """
    print(f"Received Smart Fill Request: {prompt} (Vibe: {vibe_strength})")
    
    pil_image = decode_image(await image.read())
    pil_mask = decode_image(await mask.read())
    
    result = editor.run_smart_fill(pil_image, pil_mask, prompt, vibe_strength)
    
    return Response(content=image_to_bytes(result), media_type="image/png")

if __name__ == "__main__":
    # Ensure this matches the port you exposed in Lightning Studio
    uvicorn.run(app, host="0.0.0.0", port=8080)