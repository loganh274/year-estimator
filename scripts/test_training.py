import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from src.data.loaders import get_dataloaders
from src.model.net import YearEstimator
from src.train import train_one_epoch, validate

def test_training_loop():
    print("Testing Training Loop...")
    
    # Use CPU for testing to avoid CUDA overhead if not needed, or CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a small subset of data for testing
    # We'll just use the real loaders but break early
    train_loader, val_loader = get_dataloaders(
        "data/train.csv", 
        "data/val.csv", 
        batch_size=4, 
        num_workers=0
    )
    
    # Mock loaders to return only 2 batches
    class MockLoader:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
            
        def __iter__(self):
            count = 0
            for batch in self.loader:
                yield batch
                count += 1
                if count >= 2:
                    break
        
        def __len__(self):
            return 2

    mock_train_loader = MockLoader(train_loader)
    mock_val_loader = MockLoader(val_loader)
    
    # Model
    model = YearEstimator(pretrained=False) # No need for weights
    model = model.to(device)
    
    # Optimization
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Run one epoch
    print("Running train step...")
    train_loss = train_one_epoch(model, mock_train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss}")
    
    print("Running val step...")
    val_loss = validate(model, mock_val_loader, criterion, device)
    print(f"Val Loss: {val_loss}")
    
    print("Training logic verified!")

if __name__ == "__main__":
    test_training_loop()
