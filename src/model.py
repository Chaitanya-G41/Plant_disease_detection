"""
DeiT-tiny model setup for two-stage transfer learning.

Stage 1: Fine-tune full model on 44-class PlantVillage+Guava dataset
Stage 2: Freeze early layers, replace head, fine-tune on guava only
"""

import torch
import torch.nn as nn
import timm


MODEL_NAME = 'deit_tiny_patch16_224'


def build_stage1_model(num_classes: int = 44, pretrained: bool = True) -> nn.Module:
    """
    Stage 1 model: DeiT-tiny with new classification head.
    All layers trainable — full fine-tuning on PlantVillage+Guava.

    Args:
        num_classes : total number of classes (44 in your case)
        pretrained  : load ImageNet weights
    Returns:
        model
    """
    model = timm.create_model(
        MODEL_NAME,
        pretrained  = pretrained,
        num_classes = num_classes
    )

    # Print parameter count
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model            : {MODEL_NAME}")
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Output classes   : {num_classes}")

    return model


def build_stage2_model(
    stage1_checkpoint: str,
    num_classes: int = 6,
    freeze_blocks: int = 9,
    all_classes: int = 44
) -> nn.Module:
    """
    Stage 2 model: load Stage 1 weights, freeze early transformer blocks,
    replace classification head with guava-only head.

    DeiT-tiny has 12 transformer blocks (0-11).
    Freezing blocks 0-8 (9 blocks) keeps the rich disease feature
    representations learned in Stage 1, while allowing the later
    blocks + head to specialize for guava.

    Args:
        stage1_checkpoint : path to saved Stage 1 .pth file
        num_classes       : guava classes only (6)
        freeze_blocks     : how many transformer blocks to freeze (0-11)
        all_classes       : number of classes in Stage 1 model (44)
    Returns:
        model ready for Stage 2 training
    """
    # Load Stage 1 model with original head
    model = timm.create_model(
        MODEL_NAME,
        pretrained  = False,
        num_classes = all_classes
    )
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded Stage 1 checkpoint: {stage1_checkpoint}")
    print(f"Stage 1 val accuracy: {checkpoint.get('val_acc', 'N/A')}")

    # Freeze patch embedding and positional embedding
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    model.pos_embed.requires_grad = False
    model.cls_token.requires_grad = False

    # Freeze early transformer blocks
    for i in range(freeze_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    # Replace classification head with guava-only head
    # DeiT-tiny head input size is 192
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 128),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )

    # Print what's frozen vs trainable
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\nStage 2 configuration:")
    print(f"Frozen blocks    : 0 to {freeze_blocks - 1}")
    print(f"Trainable blocks : {freeze_blocks} to 11 + new head")
    print(f"Total params     : {total:,}")
    print(f"Frozen params    : {frozen:,}  ({100*frozen/total:.1f}%)")
    print(f"Trainable params : {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"Output classes   : {num_classes}")

    return model


def get_attention_maps(model: nn.Module, x: torch.Tensor) -> list:
    """
    Extract attention maps from all transformer blocks.
    Used for attention visualization in the paper (Figure showing
    what parts of the leaf the model focuses on).

    Returns list of attention tensors, one per block.
    """
    attention_maps = []

    def hook_fn(module, input, output):
        # output shape: (batch, heads, seq_len, seq_len)
        attention_maps.append(output.detach())

    hooks = []
    for block in model.blocks:
        hook = block.attn.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        _ = model(x)

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

    return attention_maps