from typing import Tuple

import torch
from torch.types import _device


class Normalizer(torch.nn.Module):
    """Differetiable normalizer.

    Normalize input tensor without breaking computational graph.
    Input tensor might be in pixel space: [0, 255.] or unit space: [0, 1.]

    Parameters
    ----------
    input_size : int
        The size of input image which is represented by 2D tensor.
    mean : Tuple[flaot]
        The mean of input data distribution.
    std : Tuple[flaot]
        The standard diviation of input data distribution.
    device : torch.types._device
        The device used for calculation.
    from_pixcel_space : bool
        If True, an input tensor is represented in pixel space (=[0, 255.])
    """
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: _device,
        from_pixel_space: bool = True,
    ):
        super().__init__()
        self.from_pixel_space = from_pixel_space
        num_channel = len(mean)

        mean_list = [
            torch.full((input_size, input_size), mean[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)  # [1, 3, h, w]

        std_list = [
            torch.full((input_size, input_size), std[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)  # [1, 3, h, w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.from_pixel_space:
            x = x / 255.
        return x.sub(self.mean).div(self.std)


class Denormalizer(torch.nn.Module):
    """Differetiable denormalizer.

    Denormalize input tensor without breaking computational graph.
    Output tensor might be in pixel space: [0, 255.] or unit space: [0, 1.]

    Parameters
    ----------
    input_size : int
        The size of input image which is represented by 2D tensor.
    mean : Tuple[flaot]
        The mean of input data distribution.
    std : Tuple[flaot]
        The standard diviation of input data distribution.
    device : torch.types._device
        The device used for calculation.
    to_pixcel_space : bool
        If True, an output tensor is represented in pixel space (=[0, 255.])
    """
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        device: _device,
        to_pixel_space: bool = True,
    ):
        super().__init__()
        self.to_pixel_space = to_pixel_space
        num_channel = len(mean)

        mean_list = [
            torch.full((input_size, input_size), mean[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)  # [1, 3, h, w]

        std_list = [
            torch.full((input_size, input_size), std[i], device=device)
            for i in range(num_channel)
        ]  # 3 x [h, w]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)  # [1, 3, h, w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mul(self.std).add(self.mean)
        if self.to_pixel_space:
            x = x * 255.
        return x
