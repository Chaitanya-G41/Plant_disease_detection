"""
Preprocessing and augmentation transforms.
Uses ImageNet statistics — required for DeiT-tiny pretrained weights.
Augmentation strategy is conservative for val, aggressive for train
to maximize generalization on small guava classes.
"""

from torchvision import transforms


# ImageNet normalization constants (DeiT pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Args:
        mode: 'train' or 'val'
    Returns:
        torchvision transform pipeline
    """
    assert mode in ('train', 'val'), f"mode must be 'train' or 'val', got '{mode}'"

    if mode == 'train':
        return transforms.Compose([
            # Step 1: Resize slightly larger, then random crop to 224
            # This is better than direct resize — preserves more spatial detail
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMAGE_SIZE),

            # Step 2: Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),

            # Step 3: Perspective and affine — simulates different camera angles
            # Important for field conditions where farmers photograph at odd angles
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15)
            ),

            # Step 4: Color augmentations — simulate lighting variation in fields
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),

            # Step 5: Blur + noise — simulate low-quality mobile camera images
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            # Step 6: Random erasing after ToTensor
            # Forces model to look at whole leaf, not just one spot
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])

    else:  # val
        # No augmentation — only the essentials for clean evaluation
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_inverse_transform():
    """
    Inverse normalization — used for visualizing what the model actually sees.
    Useful for GradCAM / attention map overlays in the paper.
    """
    return transforms.Compose([
        transforms.Normalize(
            mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
            std=[1/s for s in IMAGENET_STD]
        )
    ])