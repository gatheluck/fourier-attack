import torch

import fourier_attack.util


class TestNormalizer:
    def test__unit_space(self,
                         cifar10_stats,
                         denormalize_cifar10_loader,
                         normalize_cifar10_loader):
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])
        mean, std = cifar10_stats
        for device in devices:
            normalizer = fourier_attack.util.Normalizer(32, mean, std, device, False)

            for (x_denorm, _), (x_norm, _) in zip(denormalize_cifar10_loader, normalize_cifar10_loader):
                x_denorm, x_norm = x_denorm.to(device), x_norm.to(device)
                assert normalizer(x_denorm).allclose(x_norm)
                break  # test only first batch

    def test__pixel_space(self,
                          cifar10_stats,
                          denormalize_cifar10_loader,
                          normalize_cifar10_loader):
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])
        mean, std = cifar10_stats
        for device in devices:
            normalizer = fourier_attack.util.Normalizer(32, mean, std, device, True)

            for (x_denorm, _), (x_norm, _) in zip(denormalize_cifar10_loader, normalize_cifar10_loader):
                x_denorm, x_norm = x_denorm.to(device), x_norm.to(device)
                assert normalizer(255. * x_denorm).allclose(x_norm, atol=1e-6)  # if atol=1e-8, allclose returns False.
                break  # test only first batch


class TestDenormalizer:
    def test__unit_space(self,
                         cifar10_stats,
                         denormalize_cifar10_loader,
                         normalize_cifar10_loader):
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])
        mean, std = cifar10_stats
        for device in devices:
            denormalizer = fourier_attack.util.Denormalizer(32, mean, std, device, False)

            for (x_denorm, _), (x_norm, _) in zip(denormalize_cifar10_loader, normalize_cifar10_loader):
                x_denorm, x_norm = x_denorm.to(device), x_norm.to(device)
                assert denormalizer(x_norm).allclose(x_denorm)
                break  # test only first batch

    def test__pixel_space(self,
                          cifar10_stats,
                          denormalize_cifar10_loader,
                          normalize_cifar10_loader):
        devices = set(["cpu", "cuda"]) if torch.cuda.is_available() else set(["cpu"])
        mean, std = cifar10_stats
        for device in devices:
            denormalizer = fourier_attack.util.Denormalizer(32, mean, std, device, True)

            for (x_denorm, _), (x_norm, _) in zip(denormalize_cifar10_loader, normalize_cifar10_loader):
                x_denorm, x_norm = x_denorm.to(device), x_norm.to(device)
                assert denormalizer(x_norm).allclose(255. * x_denorm, atol=1e-8)
                break  # test only first batch
