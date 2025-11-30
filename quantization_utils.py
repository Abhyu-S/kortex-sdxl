import torch
# FIXED IMPORT: Pointing to the specific submodule
from diffusers.quantizers.quantization_config import BitsAndBytesConfig

def get_quantization_config():
    """
    Returns the BitsAndBytes configuration for 4-bit NF4 Quantization.
    """
    print("--- Configuring 4-bit NF4 Quantization (Diffusers) ---")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )