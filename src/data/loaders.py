from torch.utils.data import DataLoader
from .dataset import YearbookDataset
from .transforms import get_transforms

def get_dataloaders(train_csv, val_csv, s3_session=None, bucket_name=None, batch_size=32, num_workers=4):
    """
    Creates and returns the training and validation DataLoaders.
    
    Args:
        train_csv (str): Path to training CSV.
        val_csv (str): Path to validation CSV.
        s3_session (boto3.Session, optional): AWS session.
        bucket_name (str, optional): S3 bucket name.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Initialize Datasets
    train_dataset = YearbookDataset(
        csv_path=train_csv,
        transform=get_transforms("train"),
        s3_session=s3_session,
        bucket_name=bucket_name
    )
    
    val_dataset = YearbookDataset(
        csv_path=val_csv,
        transform=get_transforms("val"),
        s3_session=s3_session,
        bucket_name=bucket_name
    )
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Validation block
    import torch
    import time
    
    print("Testing DataLoaders...")
    
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    
    # Use 0 workers for simple testing to avoid multiprocessing issues in simple scripts
    train_loader, val_loader = get_dataloaders(train_csv, val_csv, batch_size=16, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Fetch one batch
    start = time.time()
    images, labels = next(iter(train_loader))
    end = time.time()
    
    print(f"Batch load time: {end - start:.4f}s")
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    assert images.shape == (16, 3, 256, 256)
    assert labels.shape == (16,)
    print("Validation Successful!")
