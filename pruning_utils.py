import tomesd

def apply_token_pruning(pipeline, ratio=0.4):
    """
    Applies Token Merging (ToMe) to the SDXL UNet.
    This acts as 'Dynamic Structural Pruning', removing ~40-50% 
    of the tokens in the attention mechanism during inference.
    
    Args:
        pipeline: The Diffusers pipeline.
        ratio (float): The percentage of tokens to prune (0.0 to 1.0).
                       0.4 (40%) is a safe sweet spot for SDXL.
    """
    print(f"--- Applying Token Pruning (ToMe) with ratio {ratio} ---")
    
    # Apply ToMe patch to the UNet
    # This modifies the forward pass to merge redundant tokens
    tomesd.apply_patch(pipeline, ratio=ratio)
    
    print(f"--- Token Pruning Applied: {int(ratio*100)}% of attention tokens removed ---")
    return pipeline