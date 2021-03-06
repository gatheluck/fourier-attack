import random
from typing import Optional, Tuple

import torch

import fourier_attack.util


def get_eps(
    batch_size: int,
    eps_max: float,
    step_size_max: float,
    scale_eps: bool,
    scale_each: bool,
    device: Optional[torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """return total purturbation and single step size.

    Args:
        batch_size (int): The size of batch.
        eps_max (float): The maximum purturbation size.
        step_size_max (float): The maximum size of single step.
        scale_eps (bool): If True, randomly scale purturbation size.
        scale_each (bool): If True, scale eps independently.
        device (torch.device, optional): The device used for calculation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The tuple of eps and step_size which is used for every update.

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

    Attributes:
        device(torch.device): The device used for calculation.
        model (torch.nn.Module): The torch model in unit space.
        normalizer (fourier_attack.util.Normalizer): The normalizer from pixel space.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        device: Optional[torch.device],
    ) -> None:
        """

        Args:
            model (torch.nn.Module): The torch model in unit space.
            input_size (int): The size of input image which is represented by 2D tensor.
            mean (Tuple[float, float, float]): The mean of input data distribution.
            std (Tuple[float, float, float]): The standard diviation of input data distribution.
            device(torch.device, optional): The device used for calculation.

        """
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.normalizer = fourier_attack.util.Normalizer(
            input_size, mean, std, device=self.device
        )

    def forward(self, pixel_x: torch.Tensor) -> torch.Tensor:
        """Normalize and forward input.

        Args:
            pixel_x (torch.Tensor): The input tensor lies in unnormzlized pixel space.

        Returns:
            torch.Tensor: The output from the model.

        """
        x = self.normalizer(pixel_x)  # rescale [0, 255] -> [0, 1] and normalize
        return self.model(x)  # IMPORTANT: this return is in [0, 1]


class AttackWrapper(torch.nn.Module):
    """The wrapper of all attaker class.

    Attributes:
        input_size (int): The size of input image which is represented by 2D tensor.
        mean (Tuple[float, float, float]): The mean of input data distribution.
        std (Tuple[float, float, float]): The standard diviation of input data distribution.
        device(torch.device, optional): The device used for calculation.

    """

    def __init__(
        self,
        input_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        device: Optional[torch.device],
    ):
        """

        Args:
            input_size (int): The size of input image which is represented by 2D tensor.
            mean (Tuple[float, float, float]): The mean of input data distribution.
            std (Tuple[float, float, float]): The standard diviation of input data distribution.
            device(torch.device, optional): The device used for calculation.
            normalizer (fourier_attack.util.Normalizer): The normalizer from pixel space.
            denormalizer (fourier_attack.util.Denormalizer): The denormalizer to pixel space.

        """
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
        """Return perturbated input.

        Args:
            model (torch.nn.Module): The target NN model.
            x (torch.Tensor): The input tensor lies in normzlized unit space.

        Returns:
            torch.Tensor: The perturbed input in unit space [0,1].

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
