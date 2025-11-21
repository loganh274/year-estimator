import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model.net import YearEstimator

class YearPredictor:
    def __init__(self, model_path, device=None):
        """
        Args:
            model_path (str): Path to the saved model weights.
            device (str): 'cuda' or 'cpu'. If None, automatically detects.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        
        self.model = YearEstimator(pretrained=False) # Architecture only
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Inference transforms (same as validation)
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def predict(self, image):
        """
        Args:
            image (numpy.ndarray): RGB image.
            
        Returns:
            float: Predicted year.
        """
        # Preprocess
        if image is None:
            raise ValueError("Image is None")
            
        # Ensure RGB
        # Gradio passes RGB images, but if using cv2.imread externally it might be BGR
        # We assume input is RGB for this method
        
        augmented = self.transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device) # (1, 3, 256, 256)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            # Denormalize: pred * 100.0 + 1905
            year = output.item() * 100.0 + 1905
            
        return year

if __name__ == "__main__":
    # Validation block
    import os
    
    print("Testing YearPredictor...")
    model_path = "models/best_model.pth"
    
    if os.path.exists(model_path):
        predictor = YearPredictor(model_path)
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        year = predictor.predict(dummy_img)
        print(f"Predicted Year: {year:.2f}")
        print("Validation Successful!")
    else:
        print(f"Warning: {model_path} not found. Skipping validation.")
