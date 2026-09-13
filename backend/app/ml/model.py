from pathlib import Path
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# ImageFolder deterministically sorts directory names; keeping this order prevents
# class-index drift between training and inference.
CLASS_NAMES = ["crack", "inclusion", "normal", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
def create_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights); model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES)); return model
def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    if not path.exists(): raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(pretrained=False); model.load_state_dict(checkpoint["model_state_dict"]); model.to(device).eval()
    return model, checkpoint
