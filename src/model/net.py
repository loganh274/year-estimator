import torch
import torch.nn as nn
import timm

class YearEstimator(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_b0', pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model to use.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(YearEstimator, self).__init__()
        
        # Load backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of input features for the classifier
        # EfficientNetV2 usually has 'classifier' as the head
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            # Replace the classifier with a regression head
            self.backbone.classifier = nn.Linear(in_features, 1)
        elif hasattr(self.backbone, 'fc'): # ResNet style
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
        elif hasattr(self.backbone, 'head'): # ViT style
             in_features = self.backbone.head.in_features
             self.backbone.head = nn.Linear(in_features, 1)
        else:
            raise AttributeError("Could not find classifier/fc/head in backbone")

    def forward(self, x):
        # Output shape will be (batch_size, 1)
        return self.backbone(x)

if __name__ == "__main__":
    # Validation block
    print("Testing YearEstimator...")
    
    model = YearEstimator(pretrained=False) # No need to download weights for shape test
    
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 1)
    print("Validation Successful!")
