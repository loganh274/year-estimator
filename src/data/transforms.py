import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.
    
    Args:
        split (str): "train" or "val".
        
    Returns:
        A.Compose: The transform pipeline.
    """
    # ImageNet normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Target size as per PDD
    img_size = 256
    
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            # Optional: Add augmentations here later (e.g., HorizontalFlip, RandomBrightness)
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

if __name__ == "__main__":
    # Validation block
    import numpy as np
    import torch
    
    print("Testing Transforms...")
    
    # Create dummy image (H, W, C)
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Get transform
    transform = get_transforms("val")
    
    # Apply transform
    transformed = transform(image=dummy_img)["image"]
    
    print(f"Original shape: {dummy_img.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Transformed type: {type(transformed)}")
    
    assert transformed.shape == (3, 256, 256)
    assert isinstance(transformed, torch.Tensor)
    print("Validation Successful!")
