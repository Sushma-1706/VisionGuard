import base64
from io import BytesIO
import cv2, numpy as np, torch
from PIL import Image
from torch import nn

def gradcam(model: nn.Module, tensor: torch.Tensor, class_index: int, original: Image.Image) -> tuple[np.ndarray, str, str]:
    """Compute Grad-CAM from ResNet's final convolutional block and render evidence."""
    activations: list[torch.Tensor] = []; gradients: list[torch.Tensor] = []
    layer = model.layer4[-1].conv2
    handles = [layer.register_forward_hook(lambda _, __, out: activations.append(out)), layer.register_full_backward_hook(lambda _, gin, gout: gradients.append(gout[0]))]
    try:
        model.zero_grad(set_to_none=True); logits = model(tensor); logits[0, class_index].backward()
    finally:
        for handle in handles: handle.remove()
    weights = gradients[0].mean(dim=(2,3), keepdim=True); cam = torch.relu((weights * activations[0]).sum(dim=1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam, original.size); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    image = np.asarray(original); heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = cv2.addWeighted(image, .58, heat, .42, 0)
    def encode(array: np.ndarray) -> str:
        b = BytesIO(); Image.fromarray(array).save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
    return cam, encode(heat), encode(overlay)
