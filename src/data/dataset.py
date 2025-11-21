import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import boto3
import io

class YearbookDataset(Dataset):
    def __init__(self, csv_path, transform=None, s3_session=None, bucket_name=None):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            s3_session (boto3.Session, optional): AWS session for S3 access.
            bucket_name (str, optional): S3 bucket name.
        """
        self.bucket_name = bucket_name
        self.s3_session = s3_session
        
        # Load CSV
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        elif self.bucket_name:
            print(f"CSV {csv_path} not found locally. Attempting to load from S3...")
            try:
                if self.s3_session:
                    s3 = self.s3_session.client('s3')
                else:
                    s3 = boto3.client('s3')
                
                # Clean path for S3 key (replace backslashes)
                s3_key = csv_path.replace('\\', '/')
                
                try:
                    obj = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
                except Exception:
                    # Fallback: try stripping 'data/' prefix if it exists
                    if s3_key.startswith('data/'):
                        s3_key = s3_key[5:] # Remove 'data/'
                        obj = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
                    else:
                        raise

                self.df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                print(f"Successfully loaded {s3_key} from S3.")
            except Exception as e:
                print(f"Failed to load CSV from S3: {e}")
                raise FileNotFoundError(f"CSV not found locally or in S3: {csv_path}")
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.transform = transform
        
        # Initialize client lazily in getitem to be safe with multiprocessing
        # But we can store the credentials/config from the session to recreate it
        self.s3_client = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        img_path = row['filepath']
        year = row['year']

        image = None
        
        if self.bucket_name:
            # S3 Loading
            if self.s3_client is None:
                # Create client for this worker
                if self.s3_session:
                    self.s3_client = self.s3_session.client('s3')
                else:
                    self.s3_client = boto3.client('s3')
            
            # Handle path mapping: remove 'data/' prefix if present to match S3 keys
            # Assumption: S3 keys are like 'images/001.jpg' or 'data/images/001.jpg'
            # We try the path as is, then try stripping 'data/' if it fails (or vice versa)
            # For now, let's assume the CSV path matches the S3 key structure or is relative to root
            # If CSV has 'data/images/foo.jpg', we try that key.
            
            # Clean path for Windows compatibility if needed (replace backslashes)
            s3_key = img_path.replace('\\', '/')
            
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                image_bytes = response['Body'].read()
                # Decode
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                # Fallback: try stripping 'data/' prefix if it exists
                if s3_key.startswith('data/'):
                    try:
                        s3_key_fallback = s3_key[5:]
                        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key_fallback)
                        image_bytes = response['Body'].read()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    except Exception:
                        print(f"Error loading from S3: {s3_key} (and fallback {s3_key_fallback}) - {e}")
                        raise e
                else:
                    print(f"Error loading from S3: {s3_key} - {e}")
                    raise e
                
        else:
            # Local Loading
            image = cv2.imread(img_path)
        
        if image is None:
             raise FileNotFoundError(f"Image not found at {img_path} (S3: {self.bucket_name})")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Convert year to float32 for regression
        # Normalize: (year - 1905) / 100.0
        # Range 1905-2013 becomes 0.0 - 1.08
        norm_year = (year - 1905) / 100.0
        year_label = torch.tensor(norm_year, dtype=torch.float32)

        return image, year_label

if __name__ == "__main__":
    # Validation block
    from transforms import get_transforms
    
    print("Testing YearbookDataset...")
    
    # Use the train.csv created in Phase 2
    csv_path = "data/train.csv"
    
    if os.path.exists(csv_path):
        dataset = YearbookDataset(csv_path=csv_path, transform=get_transforms("train"))
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get first sample
        img, label = dataset[0]
        
        print(f"Image shape: {img.shape}")
        print(f"Label: {label} (type: {label.dtype})")
        
        assert img.shape == (3, 256, 256)
        assert label.dtype == torch.float32
        print("Validation Successful!")
    else:
        print(f"Warning: {csv_path} not found. Skipping validation.")
