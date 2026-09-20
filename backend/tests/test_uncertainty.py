import torch
from torch import nn
from app.ml.uncertainty import mc_dropout_probabilities
from app.ml.faithfulness import deletion_faithfulness


def test_mc_dropout_returns_probabilities_and_variance():
    model = nn.Sequential(nn.Flatten(), nn.Dropout(.5), nn.Linear(4, 2))
    mean, variance = mc_dropout_probabilities(model, torch.ones(1, 1, 2, 2), passes=5)
    assert torch.isclose(mean.sum(), torch.tensor(1.0))
    assert variance >= 0


def test_deletion_faithfulness_is_bounded():
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    deleted, random_deleted, fraction = deletion_faithfulness(model, torch.ones(1, 1, 2, 2), __import__('numpy').array([[1., 0.], [0., 0.]]), 0, .5)
    assert 0 <= deleted <= 1 and 0 <= random_deleted <= 1 and fraction == .25
