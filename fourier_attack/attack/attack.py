import random
from typing import Tuple

import torch
from torch.types import _device

import fourier_attack.util


def get_eps(
    batch_size: int,
    eps_max: float,
    step_size_max: float,
    scale_eps: bool,
    scale_each: bool,
    device: _device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    return eps and step_size (used for every update).
    """
    if scale_eps:
        # calculate scaling factor
        if scale_each:
            rand = torch.rand(batch_size, device=device)
        else:
            rand = random.random() * torch.ones(batch_size, device=device)
        # scale eps and step size
        base_eps = rand.mul(eps_max)
        step_size = rand.mul(step_size_max)
        return base_eps, step_size
    else:
        base_eps = eps_max * torch.ones(batch_size, device=device)
        step_size = step_size_max * torch.ones(batch_size, device=device)
        return base_eps, step_size


class PixelModel(torch.nn.Module):
    """Differetiable model in pixel space.

    This model takes input in unnormalized pixcel space: [0, 255.].
    Output tensor is in unit space: [0, 1.].

    Parameters
    ----------
    model : torch.nn.Module
        The torch model in unit space.
    input_size : int
        The size of input image which is represented by 2D tensor.
    mean : Tuple[flaot]
        The mean of input data distribution.
    std : Tuple[flaot]
        The standard diviation of input data distribution.
    device : torch.types._device
        The device used for calculation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: _device,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalizer = fourier_attack.util.Normalizer(
            input_size, mean, std, device=device
        )

    def forward(self, pixel_x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(pixel_x)  # rescale [0, 255] -> [0, 1] and normalize
        return self.model(x)  # IMPORTANT: this return is in [0, 1]
