import unittest

import pytest
import torch
import torchvision

import fourier_attack.attack


class TestGetEps:
    params_test_shape = {
        "scale_eps:True, scale_each:True": (True, True),
        "scale_eps:True, scale_each:False": (True, False),
        "scale_eps:False, scale_each:True": (False, True),
        "scale_eps:False, scale_each:False": (False, False),
    }

    @pytest.mark.parametrize(
        "scale_eps, scale_each",
        argvalues=list(params_test_shape.values()),
        ids=list(params_test_shape.keys()),
    )
    def test_shape(self, scale_eps, scale_each):
        batch_size = 32
        eps_max = 4.0
        step_size_max = 1.0

        # test for cpu
        assert fourier_attack.attack.get_eps(
            batch_size, eps_max, step_size_max, scale_eps, scale_each, device="cpu"
        )[0].shape == torch.Size([batch_size])
        assert fourier_attack.attack.get_eps(
            batch_size, eps_max, step_size_max, scale_eps, scale_each, device="cpu"
        )[1].shape == torch.Size([batch_size])

        # test for gpu
        if torch.cuda.is_available():
            assert fourier_attack.attack.get_eps(
                batch_size, eps_max, step_size_max, scale_eps, scale_each, device="cuda"
            )[0].shape == torch.Size([batch_size])
            assert fourier_attack.attack.get_eps(
                batch_size, eps_max, step_size_max, scale_eps, scale_each, device="cuda"
            )[1].shape == torch.Size([batch_size])

    def test_value(self):
        pass


class TestPixcelModel:
    def test__unit_space(
        self, cifar10_stats, denormalize_cifar10_loader, normalize_cifar10_loader
    ):
        model = torchvision.models.resnet50(pretrained=False, num_classes=10).eval()
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])
        mean, std = cifar10_stats
        for device in devices:
            model = model.to(device)
            pixel_model = fourier_attack.attack.PixelModel(model, 32, mean, std, device)

            for (x_denorm, _), (x_norm, _) in zip(
                denormalize_cifar10_loader, normalize_cifar10_loader
            ):
                x_denorm, x_norm = x_denorm.to(device), x_norm.to(device)
                assert pixel_model(255.0 * x_denorm).allclose(
                    model(x_norm), atol=1e-5
                )  # if atol=1e-8, allclose returns False.
                break


class TestAttackWrapper:
    def test__forward(self, cifar10_stats):
        batch_size = 8
        input_size = 32
        mean, std = cifar10_stats

        model = torchvision.models.resnet50(pretrained=False, num_classes=10).eval()
        input_sample = torch.randn(
            batch_size, 3, input_size, input_size, dtype=torch.float
        )
        return_sample = torch.randn(
            batch_size, 3, input_size, input_size, dtype=torch.float
        )
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])

        for device in devices:

            def _forward_mock(self, *args, **kwargs):
                return return_sample.to(device)

            model, input_sample = model.to(device), input_sample.to(device)

            with unittest.mock.patch.object(
                fourier_attack.attack.AttackWrapper, "_forward", _forward_mock
            ):
                attacker = fourier_attack.attack.AttackWrapper(
                    input_size, mean, std, device
                )
                assert attacker(model, input_sample).shape == torch.Size(
                    [batch_size, 3, input_size, input_size]
                )
