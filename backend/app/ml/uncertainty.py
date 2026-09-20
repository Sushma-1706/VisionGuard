import torch
from torch import nn


def mc_dropout_probabilities(model: nn.Module, tensor: torch.Tensor, passes: int = 20) -> tuple[torch.Tensor, float]:
    """Return mean class probabilities and variance of the winning-class probability."""
    was_training = model.training
    model.eval()
    dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    for layer in dropout_layers:
        layer.train()
    try:
        with torch.no_grad():
            samples = torch.stack([torch.softmax(model(tensor), dim=1)[0] for _ in range(passes)])
    finally:
        model.train(was_training)
    mean = samples.mean(dim=0)
    winner = int(mean.argmax())
    return mean, float(samples[:, winner].var(unbiased=False).item())
