import torch
import torch.nn as nn
from torchvision import models

class VisualEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(VisualEmotionModel, self).__init__()
        # Load EfficientNet-B0 with default weights
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        except AttributeError:
             # Fallback for older torchvision versions
            self.backbone = models.efficientnet_b0(pretrained=True)

        # The efficientnet_b0 classifier is:
        # Sequential(
        #   (0): Dropout(p=0.2, inplace=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        # We replace the Linear layer to match our number of classes.
        
        # Get the input features of the final linear layer
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def get_embedding(self, x):
        """
        Extracts the 1280-d embedding before the classifier.
        """
        # EfficientNet implementation details for feature extraction
        # .features() gives the convolutional features
        x = self.backbone.features(x)
        # Global Average Pooling
        x = self.backbone.avgpool(x)
        # Flatten
        x = torch.flatten(x, 1)
        return x
