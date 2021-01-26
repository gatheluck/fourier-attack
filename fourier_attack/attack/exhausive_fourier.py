from typing import Callable, Dict, Final, Tuple, Union

import torch
import torch.nn
from torch.types import _device

import fourier_attack.attack
import fourier_attack.fourier.basis
import fourier_attack.util


class ExhausiveFourierAttack(fourier_attack.attack.AttackWrapper):
    """Exhausive fourier attack.

    This model takes input in unnormalized pixcel space: [0, 255.].
    Output tensor is in unit space: [0, 1.].

    Parameters
    ----------
    model : torch.nn.Module
        The torch model in unit space.
    input_size : int
        The size of input image which is represented by 2D tensor.
    mean : Tuple[float]
        The mean of input data distribution.
    std : Tuple[float]
        The standard diviation of input data distribution.
    eps_max : float
        The max size of purturbation.
    criterion_func : Callable[..., torch.Tensor]
        The loss function used for training model.
    device : torch.types._device
        The device used for calculation.
    """

    def __init__(
        self,
        input_size: int,
        mean: Tuple[float],
        std: Tuple[float],
        eps_max: float,
        criterion_func: Callable[..., torch.Tensor],
        device: Union[_device, str, None],
    ) -> None:
        super().__init__(input_size=input_size, mean=mean, std=std, device=device)
        self.eps_max = eps_max
        self.criterion_func = criterion_func
        self.device = device

    def _forward(
        self,
        pixel_model: torch.nn.Module,
        pixel_x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """"""
        batch_size: Final[int] = pixel_x.size(0)

        eps = self.eps_max * torch.ones(batch_size).to(self.device)

        sign_plus = (
            torch.ones(batch_size)[:, None].repeat(1, 3).to(self.device)
        )  # (B, 3)
        channel_sign = torch.where(
            torch.rand_like(sign_plus) >= 0.5, sign_plus, -sign_plus
        )

        pixel_perturbation, return_dict = self.run(
            pixel_model, pixel_x, target, eps, channel_sign, self.criterion_func
        )

        # IMPORTANT: this return is in PIXEL SPACE (=[0,255])
        return pixel_x + pixel_perturbation

    @classmethod
    def run(
        cls,
        pixel_model: torch.nn.Module,
        pixel_x: torch.Tensor,
        target: torch.Tensor,
        eps: torch.Tensor,
        channel_sign: torch.Tensor,
        criterion_func: Callable[..., torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        pixel_model : torch.nn.Module
            The pixel model
        pixel_x : torch.Tensor, (B,C,H,W)
            The input tensor in pixel space.
        target : torch.Tensor, (B)
            The target label of input.
        eps : torch.Tensor, (B)
            The size of purturbation.
        channel_sign : torch.Tensor, (B,C)
            The sign of purturbation.
        criterion_func : Callable
            The criterion function.
        """
        B, _, H, W = pixel_x.size()
        W_ = W // 2 + 1
        F = H * W_
        device: Final = pixel_x.device

        with torch.no_grad():
            losses = torch.zeros(B, F, dtype=torch.float, device=device)  # (B, F)
            corrects = torch.zeros(B, F, dtype=torch.float, device=device)  # (B, F)

            all_spectrum = fourier_attack.fourier.basis.get_basis_spectrum(H, W_).to(
                device
            )  # (F, H, W_)
            all_pixel_basis = 255.0 * fourier_attack.fourier.basis.spectrum_to_basis(
                all_spectrum
            )  # (F, H, W)

            for f in range(F):
                pixel_basis = all_pixel_basis[f, :, :]  # (H, W)
                pixel_basis_ = pixel_basis[None, None, :, :].repeat(
                    B, 3, 1, 1
                )  # (B, 3, H, W)
                pixel_basis_ *= (
                    eps[:, None, None, None] * channel_sign[:, :, None, None]
                )

                pixel_x_adv = torch.clamp(
                    pixel_x + pixel_basis_, 0.0, 255.0
                )  # (B, 3, H, W)
                logit = pixel_model(pixel_x_adv)  # (B, #class)

                loss = criterion_func(logit, target, reduction="none")  # (B)
                losses[:, f] = loss

                correct = fourier_attack.util.get_corrects(logit, target)[0]
                corrects[:, f] = correct

            index = losses.argmax(dim=-1)[:, None, None, None].repeat(
                1, 1, H, W
            )  # (B, 1, H, W)
            pixel_basis_adv = (
                all_pixel_basis[None, :, :, :]
                .repeat(B, 1, 1, 1)
                .gather(dim=1, index=index)
                .view(B, 1, H, W)
            )
            pixel_basis_adv_ = pixel_basis_adv.repeat(1, 3, 1, 1)
            pixel_basis_adv_ *= (
                eps[:, None, None, None] * channel_sign[:, :, None, None]
            )

        # calculate returns
        pixel_perturbation = (
            torch.clamp(pixel_x + pixel_basis_adv_, 0.0, 255.0) - pixel_x
        )

        return_dict = {
            "losses": losses.view(B, H, W_),
            "corrects": corrects.view(B, H, W_),
        }
        return pixel_perturbation, return_dict
