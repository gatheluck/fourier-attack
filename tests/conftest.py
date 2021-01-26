import pathlib
import random
from typing import Final

import numpy as np
import pytest
import torch
import torchvision

CIFAR10_ROOT: Final = pathlib.Path("data/cifar10")
PRETRAONED_WEIGHT = pathlib.Path("tests/weight/cifar10_resnet50.pth")
PRETRAONED_WEIGHT_ADV = pathlib.Path("tests/weight/cifar10_resnet50_adv.pth")
CIFAR10_MEAN: Final = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD: Final = [0.24703223, 0.24348513, 0.26158784]
BATCH_SIZE = 16

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


@pytest.fixture
def cifar10_stats():
    return CIFAR10_MEAN, CIFAR10_STD


@pytest.fixture
def normalize_cifar10_loader():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        str(CIFAR10_ROOT), False, transform, download=True
    )
    return torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8
    )


@pytest.fixture
def denormalize_cifar10_loader():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        str(CIFAR10_ROOT), False, transform, download=True
    )
    return torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8
    )


@pytest.fixture
def pretrained_cifar10_resnet50():
    model = torchvision.models.resnet50(pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(PRETRAONED_WEIGHT))
    return model.eval()


@pytest.fixture
def pretrained_cifar10_resnet50_adv():
    model = torchvision.models.resnet50(pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(PRETRAONED_WEIGHT_ADV))
    return model.eval()
