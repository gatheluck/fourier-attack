import pathlib
from typing import Final

import torch
import torchvision

import fourier_attack.attack.exhausive_fourier


class TestExhausiveFourierAttack:
    def test__run(
        self, pretrained_cifar10_resnet50, cifar10_stats, denormalize_cifar10_loader
    ):
        devices = {"cuda"}
        output_root: Final = pathlib.Path("logs/test/")
        output_root.mkdir(exist_ok=True)

        model = pretrained_cifar10_resnet50
        criterion_func = torch.nn.functional.cross_entropy

        mean, std = cifar10_stats
        for device in devices:
            pixel_model = fourier_attack.attack.PixelModel(model, 32, mean, std, device)

            for x_denorm, t in denormalize_cifar10_loader:
                x_denorm, t = x_denorm.to(device), t.to(device)
                pixel_x = 255.0 * x_denorm

                batch_size = pixel_x.size(0)
                eps = 8.0 * torch.ones(batch_size).to(device)

                sign_plus = torch.ones(batch_size)[:, None].repeat(1, 3).to(device)
                channel_sign = torch.where(
                    torch.rand_like(sign_plus) >= 0.5, sign_plus, -sign_plus
                )

                (
                    perb,
                    return_dict,
                ) = fourier_attack.attack.exhausive_fourier.ExhausiveFourierAttack.run(
                    pixel_model, pixel_x, t, eps, channel_sign, criterion_func
                )

                x_adv = (pixel_x + perb) / 255.0
                perb_ = (perb / 255.0) + 0.5
                torchvision.utils.save_image(
                    x_adv, output_root / "run_exhausive_fourier.png"
                )
                torchvision.utils.save_image(
                    perb_, output_root / "run_perturbation.png"
                )

                losses = return_dict["losses"]  # (B, H, W_)
                losses /= torch.max(losses.view(batch_size, -1), dim=-1)[0][
                    :, None, None
                ]
                torchvision.utils.save_image(
                    losses[:, None, :, :].repeat(1, 3, 1, 1),
                    output_root / "run_losses.png",
                )

                errors = (
                    torch.ones_like(return_dict["corrects"]) - return_dict["corrects"]
                )
                torchvision.utils.save_image(
                    errors[:, None, :, :].repeat(1, 3, 1, 1),
                    output_root / "run_errors.png",
                )

                assert perb.size() == torch.Size([batch_size, 3, 32, 32])
                break  # test only first batch

    def test__forward(
        self, pretrained_cifar10_resnet50, cifar10_stats, normalize_cifar10_loader
    ):
        input_size: Final[int] = 32
        eps_max: Final[float] = 8.0
        devices = {"cuda"}
        output_root: Final = pathlib.Path("logs/test/")
        output_root.mkdir(exist_ok=True)

        model = pretrained_cifar10_resnet50
        criterion_func = torch.nn.functional.cross_entropy

        mean, std = cifar10_stats
        for device in devices:
            attacker = fourier_attack.attack.exhausive_fourier.ExhausiveFourierAttack(
                input_size, mean, std, eps_max, criterion_func, device
            )

            for x, t in normalize_cifar10_loader:
                x, t = x.to(device), t.to(device)
                batch_size = x.size(0)

                x_adv = attacker(model, x, t)
                torchvision.utils.save_image(
                    x_adv, output_root / "forward_exhausive_fourier.png"
                )

                assert x_adv.size() == torch.Size([batch_size, 3, 32, 32])
                break  # test only first batch
