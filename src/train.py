import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import boto3
import botocore.exceptions

from src.data.loaders import get_dataloaders
from src.model.net import YearEstimator

def get_aws_credentials():
    """
    Interactively prompts for AWS credentials if not found in environment variables.
    Validates connection to the S3 bucket.
    """
    print("\n=== AWS S3 Configuration ===")
    
    # 1. Get Credentials
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_REGION', 'us-east-1')
    bucket_name = os.environ.get('S3_BUCKET_NAME')

    if not all([access_key, secret_key, bucket_name]):
        print("AWS credentials not found in environment variables.")
        print("Please enter your AWS details (these will not be saved to disk):")
        
        if not access_key:
            access_key = input("AWS Access Key ID: ").strip()
        if not secret_key:
            secret_key = input("AWS Secret Access Key: ").strip()
        if not bucket_name:
            bucket_name = input("S3 Bucket Name: ").strip()
        
        region_input = input(f"AWS Region [{region}]: ").strip()
        if region_input:
            region = region_input

    # 2. Create Session
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    # 3. Validate Connection
    print(f"\nConnecting to bucket '{bucket_name}'...")
    s3 = session.client('s3', config=botocore.config.Config(connect_timeout=90, read_timeout=90))
    
    try:
        # Attempt to list a single object to verify permissions and connectivity
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print("✅ Connection successful!")
        return session, bucket_name
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            print("❌ Error: Access Denied (403). Please check your credentials and bucket permissions.")
        elif error_code == '404':
            print(f"❌ Error: Bucket '{bucket_name}' not found (404).")
        else:
            print(f"❌ AWS Client Error: {e}")
        raise e
    except botocore.exceptions.ConnectTimeout:
        print("❌ Error: Connection timed out (90s). Please check your internet connection.")
        raise
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        raise

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # (batch, 1)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate MAE (Mean Absolute Error) - same as L1Loss but good to be explicit
    # In this case criterion is L1Loss so epoch_loss IS MAE.
    
    return epoch_loss

def main():
    print("Initializing training script...")
    
    # AWS Setup
    try:
        s3_session, bucket_name = get_aws_credentials()
    except Exception:
        print("Failed to initialize AWS connection. Exiting.")
        return

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4 # Increased from 0 to 4 to improve GPU utilization
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15 # Increased for better convergence with normalization
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        "data/train.csv", 
        "data/val.csv", 
        s3_session=s3_session,
        bucket_name=bucket_name,
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    # Model
    model = YearEstimator(pretrained=True)
    model = model.to(DEVICE)
    
    # Optimization
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    best_val_loss = float('inf')
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Time: {epoch_mins:.2f}m "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model with Val Loss: {val_loss:.4f}")
            
    print("Training complete.")

if __name__ == "__main__":
    main()
