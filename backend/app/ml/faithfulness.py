import numpy as np
import torch


def deletion_faithfulness(model, tensor: torch.Tensor, cam: np.ndarray, class_index: int, threshold: float) -> tuple[float, float, float]:
    """Compare confidence loss after masking salient pixels with a random mask."""
    mask = torch.from_numpy((cam >= threshold).astype(np.bool_)).to(tensor.device)
    mask = torch.nn.functional.interpolate(mask[None, None].float(), size=tensor.shape[-2:], mode="nearest").bool()[0, 0]
    fraction = float(mask.float().mean())
    if fraction == 0:
        return 0.0, 0.0, 0.0
    generator = torch.Generator(device=tensor.device).manual_seed(7)
    random_mask = torch.rand(mask.shape, generator=generator, device=tensor.device) < fraction
    baseline = torch.zeros_like(tensor)
    with torch.no_grad():
        original = torch.softmax(model(tensor), dim=1)[0, class_index]
        deleted = torch.softmax(model(torch.where(mask[None, None], baseline, tensor)), dim=1)[0, class_index]
        random_deleted = torch.softmax(model(torch.where(random_mask[None, None], baseline, tensor)), dim=1)[0, class_index]
    return float((original - deleted).clamp_min(0)), float((original - random_deleted).clamp_min(0)), fraction
