import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import boto3
import botocore.exceptions
import botocore.config

from src.data.loaders import get_dataloaders
from src.model.net import YearEstimator

def get_aws_credentials(access_key=None, secret_key=None, bucket_name=None, region='us-east-1'):
    """
    Validates connection to the S3 bucket using provided credentials or environment variables.
    """
    # 1. Get Credentials (if not provided)
    if not access_key:
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    if not secret_key:
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not bucket_name:
        bucket_name = os.environ.get('S3_BUCKET_NAME')
    
    if not region:
        region = os.environ.get('AWS_REGION', 'us-east-1')

    if not all([access_key, secret_key, bucket_name]):
        # Interactive CLI Fallback
        print("AWS credentials not found.")
        print("Please enter your AWS details:")
        
        if not access_key:
            access_key = input("AWS Access Key ID: ").strip()
        if not secret_key:
            secret_key = input("AWS Secret Access Key: ").strip()
        if not bucket_name:
            bucket_name = input("S3 Bucket Name: ").strip()
        
        region_input = input(f"AWS Region [{region}]: ").strip()
        if region_input:
            region = region_input

    # Ensure no whitespace
    if access_key: access_key = access_key.strip()
    if secret_key: secret_key = secret_key.strip()
    if bucket_name: bucket_name = bucket_name.strip()
    if region: region = region.strip()

    if not all([access_key, secret_key, bucket_name]):
        raise ValueError("Missing AWS credentials. Please provide them.")

    # 2. Create Session
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    # 3. Validate Connection
    # print(f"\nConnecting to bucket '{bucket_name}'...") # Let caller handle logging
    s3 = session.client('s3', config=botocore.config.Config(connect_timeout=90, read_timeout=90))
    
    try:
        # Attempt to list a single object to verify permissions and connectivity
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        return session, bucket_name
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            raise PermissionError("Access Denied (403). Check credentials.")
        elif error_code == '404':
            raise FileNotFoundError(f"Bucket '{bucket_name}' not found.")
        else:
            raise e
    except botocore.exceptions.ConnectTimeout:
        raise TimeoutError("Connection timed out.")
    except Exception as e:
        raise e

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

def run_training(access_key, secret_key, bucket_name, log_callback=print):
    """
    Main entry point for training, callable from GUI.
    """
    log_callback("Initializing training script...")
    
    # AWS Setup
    try:
        s3_session, bucket_name = get_aws_credentials(access_key, secret_key, bucket_name)
        log_callback("✅ AWS Connection successful!")
    except Exception as e:
        log_callback(f"❌ Failed to initialize AWS connection: {str(e)}")
        return

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Set to 0 for Windows GUI compatibility to avoid multiprocessing issues
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_callback(f"Using device: {DEVICE}")
    
    try:
        # Data
        log_callback("Setting up data loaders (this may take a moment)...")
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
        
        log_callback("Starting training...")
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            
            # TODO: Pass log_callback to train_one_epoch if we want per-batch updates
            # For now, we'll just log per epoch to keep it clean in the GUI
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = validate(model, val_loader, criterion, DEVICE)
            
            end_time = time.time()
            epoch_mins = (end_time - start_time) / 60
            
            log_msg = (f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                       f"Time: {epoch_mins:.2f}m "
                       f"Train Loss: {train_loss:.4f} "
                       f"Val Loss: {val_loss:.4f}")
            log_callback(log_msg)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                log_callback(f"  >> Saved best model with Val Loss: {val_loss:.4f}")
                
        log_callback("Training complete.")
        
    except Exception as e:
        log_callback(f"❌ An error occurred during training: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())

def main():
    # CLI entry point
    print("=== CLI Training Mode ===")
    try:
        # For CLI, we still want to support env vars or input, but get_aws_credentials 
        # now expects args or env vars. We can wrap it or just call run_training with None
        # and let it fall back to env vars.
        run_training(None, None, None)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

if __name__ == "__main__":
    main()
