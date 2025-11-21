import sys
import os
import time

# Add src to path so we can import modules
sys.path.append(os.path.abspath("src"))

from data.loaders import get_dataloaders

def test_loaders():
    print("Testing DataLoaders...")
    
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    
    # Use 0 workers for simple testing
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

if __name__ == "__main__":
    test_loaders()
