import random
from typing import Tuple

import torch
from torch.types import _device


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
