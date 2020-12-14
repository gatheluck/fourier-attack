import pytest
import torch

import fourier_attack.fourier.basis


class TestSpectrumToBasis:
    @pytest.fixture
    def sample_spectrum(self):
        B, H, W = 16, 32, 17
        return torch.rand(B, H, W)

    def test__output_shape(self, sample_spectrum):
        B, H, W = sample_spectrum.size()
        assert fourier_attack.fourier.basis.spectrum_to_basis(
            sample_spectrum, False
        ).size() == torch.Size([B, H, H])
        assert fourier_attack.fourier.basis.spectrum_to_basis(
            sample_spectrum, True
        ).size() == torch.Size([B, H, H])

    def test__norm(self, sample_spectrum):
        B, _, _ = sample_spectrum.size()
        ones = torch.ones(B, dtype=torch.float)
        assert not torch.allclose(
            fourier_attack.fourier.basis.spectrum_to_basis(sample_spectrum, False).norm(
                dim=(-2, -1)
            ),
            ones,
        )
        assert torch.allclose(
            fourier_attack.fourier.basis.spectrum_to_basis(sample_spectrum, True).norm(
                dim=(-2, -1)
            ),
            ones,
        )
