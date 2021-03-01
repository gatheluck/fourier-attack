import random
from typing import Callable, Final, Optional, Tuple

import torch
import torch.nn

import fourier_attack.attack
import fourier_attack.fourier.basis
import fourier_attack.util


def normalized_random_init(
    shape: torch.Size, norm: str, device: torch.device
) -> torch.Tensor:
    """
    Args:
        shape: shape of expected tensor. eg.) (B,C,H,W)
        norm: type of norm
    """
    if norm == "linf":
        init = (
            2.0 * torch.rand(shape, dtype=torch.float, device=device) - 1.0
        )  # values are in [-1, +1]
    elif norm == "l2":
        init = 2.0 * torch.randn(
            shape, dtype=torch.float, device=device
        )  # values in init are sampled form N(0,1)
        init_norm = torch.norm(init.view(init.size(0), -1), p=2.0, dim=1)  # (B)
        normalized_init = init / init_norm[:, None, None, None]

        dim = init.size(1) * init.size(2) * init.size(3)
        rand_norm = torch.pow(
            torch.rand(init.size(0), dtype=torch.float, device=device), 1.0 / dim
        )
        init = normalized_init * rand_norm[:, None, None, None]
    else:
        raise NotImplementedError

    return init


class PgdAttack(fourier_attack.attack.AttackWrapper):
    def __init__(
        self,
        input_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        num_iteration: int,
        eps_max: float,
        step_size: float,
        norm: str,
        rand_init: bool,
        scale_eps: bool,
        scale_each: bool,
        avoid_target: bool,
        criterion_func: Callable[..., torch.Tensor],
        device: Optional[torch.device],
    ) -> None:
        """
        """
        super().__init__(input_size=input_size, mean=mean, std=std, device=device)
        self.num_iteration = num_iteration
        self.eps_max = eps_max
        self.step_size = step_size
        self.norm = norm
        self.rand_init = rand_init
        self.scale_eps = scale_eps
        self.scale_each = scale_each
        self.avoid_target = avoid_target
        self.criterion_func = criterion_func

    def _forward(
        self,
        pixel_model: torch.nn.Module,
        pixel_x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return perturbed input by PGD in pixel space [0,255]

        Note:
            This method shoud be called through fourier_attack.attack.AttackWrapper.forward method.
            DO NOT call this method directly.

        Args:
            pixel_model (torch.nn.Module):  The torch model in pixel space.
            pixel_x (torch.Tensor): The input tensor lies in unnormzlized pixel space.
            target (torch.Tensor): The target labels of pixel_x.

        Returns:
            torch.Tensor: The perturbed input in pixel space [0,255].

        """
        # if scale_eps is True, change eps adaptively.
        # this process usually improves robustness against wide range of attack.
        batch_size: Final = pixel_x.size(0)
        base_eps, step_size = self.get_base_eps_and_step_size(batch_size)

        # init delta
        pixel_input = pixel_x.detach()
        pixel_input.requires_grad_()
        pixel_delta = self._init_delta(pixel_input.size(), base_eps)  # (B,C,H,W)

        # compute delta in pixel space
        if self.num_iteration:  # run iteration
            pixel_delta = self._run(
                pixel_model, pixel_input, pixel_delta, target, base_eps, step_size,
            )
        else:  # if self.num_iteration is 0, return just initialization result
            pixel_delta.data = (
                torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                - pixel_input.data
            )

        # IMPORTANT: this return is in PIXEL SPACE (=[0,255])
        return pixel_input + pixel_delta

    def get_base_eps_and_step_size(
            self,
            batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method return appropriate base_eps and step_size.

        Args:
            batch_size (int): The size of batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple of tensors; base_eps and step_size whose size are (B).

        """
        if self.scale_eps:
            if self.scale_each:
                rand = torch.rand(batch_size, device=self.device)  # (B)
            else:
                rand = random.random() * torch.ones(batch_size, device=self.device)
            return rand.mul(self.eps_max), rand.mul(self.step_size)
        else:
            base_eps = self.eps_max * torch.ones(batch_size, device=self.device)  # (B)
            step_size = self.step_size * torch.ones(batch_size, device=self.device)  # (B)
            return base_eps, step_size

    def _init_delta(self, shape: torch.Size, eps: torch.Tensor) -> torch.Tensor:
        """Initialize delta. If self.rand_init is True, execute random initialization.

        Args:
            shape (torch.Size):
            eps (torch.Tensor):

        Returns:
            torch.Tensor: Initial delta (perturbation) shape of (B,C,H,W)

        """
        # If self.rand_init is Flase, initialize by zero.
        if not self.rand_init:
            return torch.zeros(shape, requires_grad=True, device=self.device)
        else:
            init_delta = normalized_random_init(shape, self.norm, self.device)  # initialize delta for linf or l2
            init_delta = eps[:, None, None, None] * init_delta  # scale by eps
            init_delta.requires_grad_()
            return init_delta

    def _run(
        self,
        pixel_model: torch.nn.Module,
        pixel_input: torch.Tensor,
        pixel_delta: torch.Tensor,
        target: torch.Tensor,
        eps: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        """Run iterations of PGD attack.

        Args:
            pixel_model (torch.nn.Module):  The torch model in pixel space.
            pixel_input (torch.Tensor): The input tensor lies in unnormzlized pixel space.
            pixel_delta (torch.Tensor): The initial purterbation lies in unnormzlized pixel space.
            target (torch.Tensor): The target labels of pixel_x.
            eps (torch.Tensor): The size of purturbation.
            step_size (torch.Tensor): The size of single iteration step.

        Returns:
            torch.Tensor: The perturbation (delta) in pixel space [0,255].

        """
        logit = pixel_model(pixel_input + pixel_delta)
        if self.norm == "l2":
            l2_eps_max = eps

        for it in range(self.num_iteration):
            loss = self.criterion_func(logit, target)
            loss.backward()

            if self.avoid_target:
                grad = pixel_delta.grad.data  # to avoid target, increase the loss
            else:
                grad = -pixel_delta.grad.data  # to hit target, decrease the loss

            if self.norm == "linf":
                grad_sign = grad.sign()
                pixel_delta.data = (
                    pixel_delta.data + step_size[:, None, None, None] * grad_sign
                )
                pixel_delta.data = torch.max(
                    torch.min(pixel_delta.data, eps[:, None, None, None]),
                    -eps[:, None, None, None],
                )  # scale in [-eps, +eps]
                pixel_delta.data = (
                    torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                    - pixel_input.data
                )
            elif self.norm == "l2":
                batch_size = pixel_delta.data.size(0)
                grad_norm = torch.norm(
                    grad.view(batch_size, -1), p=2.0, dim=1
                )  # IMPORTANT: if you set eps = 0.0 this leads nan
                normalized_grad = grad / grad_norm[:, None, None, None]
                pixel_delta.data = (
                    pixel_delta.data + step_size[:, None, None, None] * normalized_grad
                )
                l2_pixel_delta = torch.norm(
                    pixel_delta.data.view(batch_size, -1), p=2.0, dim=1
                )
                # check numerical instabitily
                proj_scale = torch.min(
                    torch.ones_like(l2_pixel_delta, device=self.device),
                    l2_eps_max / l2_pixel_delta,
                )
                pixel_delta.data = pixel_delta.data * proj_scale[:, None, None, None]
                pixel_delta.data = (
                    torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0)
                    - pixel_input.data
                )
            else:
                raise NotImplementedError

            if it != self.num_iteration - 1:
                logit = pixel_model(pixel_input + pixel_delta)
                pixel_delta.grad.data.zero_()

        return pixel_delta
