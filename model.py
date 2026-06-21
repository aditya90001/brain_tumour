import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
    )

    return model