import random
from typing import Tuple, Union

import torch
from torch.types import Device

import fourier_attack.util


def get_eps(
    batch_size: int,
    eps_max: float,
    step_size_max: float,
    scale_eps: bool,
    scale_each: bool,
    device: Device,
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
        device: Device,
    ) -> None:
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.normalizer = fourier_attack.util.Normalizer(
            input_size, mean, std, device=self.device
        )

    def forward(self, pixel_x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(pixel_x)  # rescale [0, 255] -> [0, 1] and normalize
        return self.model(x)  # IMPORTANT: this return is in [0, 1]


class AttackWrapper(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: Device,
    ):
        super().__init__()
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.device = device
        self.normalizer = fourier_attack.util.Normalizer(
            self.input_size, self.mean, self.std, device=self.device
        )
        self.denormalizer = fourier_attack.util.Denormalizer(
            self.input_size, self.mean, self.std, device=self.device
        )

    def forward(
        self, model: torch.nn.Module, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Return perturbed input in unit space [0,1]
        This function shold be called from all Attacker.
        """
        was_training = model.training
        pixel_model = PixelModel(
            model, self.input_size, self.mean, self.std, self.device
        )
        pixel_model.eval()
        # forward input to  pixel space
        pixel_x = self.denormalizer(x.detach())
        pixel_return = self._forward(pixel_model, pixel_x, *args, **kwargs)  # type: ignore
        if was_training:
            pixel_model.train()

        return self.normalizer(pixel_return)
