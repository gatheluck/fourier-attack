import pytest
import torch

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
