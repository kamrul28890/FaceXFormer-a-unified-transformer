"""
Count total parameters in FaceXFormer-main model.
"""

import sys
import torch
sys.path.append('network/models')

from network.models.facexformer import FaceXFormer


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_module_parameters(model):
    """Get parameter counts for each module."""
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    return module_params


if __name__ == "__main__":
    print("="*70)
    print("FaceXFormer-main Parameter Count")
    print("="*70)
    
    # Create model
    print("\nInitializing model...")
    model = FaceXFormer()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{'='*70}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"{'='*70}")
    
    # Get module-wise breakdown
    print("\nModule-wise Parameter Breakdown:")
    print("-"*70)
    module_params = get_module_parameters(model)
    
    for name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (params / total_params) * 100
        print(f"{name:30s}: {params:>15,} ({percentage:>5.2f}%)")
    
    # Get more detailed breakdown for key components
    print(f"\n{'='*70}")
    print("Detailed Component Breakdown:")
    print("-"*70)
    
    # Backbone
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"{'Backbone (Swin-B)':30s}: {backbone_params:>15,} ({backbone_params/total_params*100:>5.2f}%)")
    
    # MLPs
    mlp_params = sum(p.numel() for p in model.linear_c.parameters())
    print(f"{'Level MLPs':30s}: {mlp_params:>15,} ({mlp_params/total_params*100:>5.2f}%)")
    
    # Linear fuse
    fuse_params = sum(p.numel() for p in model.linear_fuse.parameters())
    print(f"{'Linear Fuse':30s}: {fuse_params:>15,} ({fuse_params/total_params*100:>5.2f}%)")
    
    # Face decoder
    decoder_params = sum(p.numel() for p in model.face_decoder.parameters())
    print(f"{'Face Decoder':30s}: {decoder_params:>15,} ({decoder_params/total_params*100:>5.2f}%)")
    
    # PE layer
    pe_params = sum(p.numel() for p in model.pe_layer.parameters())
    print(f"{'Position Encoding':30s}: {pe_params:>15,} ({pe_params/total_params*100:>5.2f}%)")
    
    print(f"\n{'='*70}")
    
    # Memory estimate
    param_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
    print(f"\nEstimated Model Size (float32): {param_size_mb:.2f} MB")
    print(f"Estimated Model Size (float16): {param_size_mb/2:.2f} MB")
    
    print(f"\n{'='*70}\n")
