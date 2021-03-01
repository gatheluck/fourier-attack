import pathlib
from typing import Final

import torch
import torchvision

import fourier_attack.attack.pgd
from fourier_attack.util import Denormalizer


class TestPgdAttack:
    def test__forward(
        self, pretrained_cifar10_resnet50, cifar10_stats, normalize_cifar10_loader
    ):
        input_size: Final = 32
        num_iteration: Final = 8
        eps_max: Final = 16.0
        step_size: Final = eps_max / num_iteration
        rand_init: Final = True
        scale_eps: Final = True
        scale_each: Final = True
        avoid_target: Final = True
        norms = {"linf", "l2"}
        devices = set(["cuda"]) if torch.cuda.is_available() else set()
        output_root: Final = pathlib.Path("logs/test/")
        output_root.mkdir(exist_ok=True, parents=True)

        model = pretrained_cifar10_resnet50
        criterion_func = torch.nn.functional.cross_entropy

        mean, std = cifar10_stats
        for norm in norms:
            for device in devices:
                attacker = fourier_attack.attack.pgd.PgdAttack(
                    input_size,
                    mean,
                    std,
                    num_iteration,
                    eps_max,
                    step_size,
                    norm,
                    rand_init,
                    scale_eps,
                    scale_each,
                    avoid_target,
                    criterion_func,
                    device,
                )

                for x, t in normalize_cifar10_loader:
                    x, t = x.to(device), t.to(device)
                    batch_size = x.size(0)

                    x_adv = attacker(model, x, t)
                    denormalizer = Denormalizer(input_size, mean, std, device, False)
                    torchvision.utils.save_image(
                        denormalizer(x_adv), output_root / f"forward-pgd-{norm}.png"
                    )

                    assert x_adv.size() == torch.Size([batch_size, 3, 32, 32])
                    break  # test only first batch
