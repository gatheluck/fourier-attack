from typing import Final

import pytest
import torch

import fourier_attack.util


class TestGetCorrects:
    @pytest.fixture
    def sample_output(self):
        return torch.tensor(
            [
                [1.0, 0.9, 0.8, 0.7, 0.6],
                [0.1, 1.0, 0.9, 0.8, 0.7],
                [0.1, 0.1, 1.0, 0.9, 0.8],
                [0.1, 0.1, 0.1, 1.0, 0.9],
                [0.1, 0.1, 0.1, 0.1, 1.0],
            ],
            dtype=torch.float,
        )

    @pytest.fixture
    def sample_target(self):
        return torch.tensor([4, 4, 4, 4, 4], dtype=torch.long)

    ans_top1: Final = torch.tensor([0, 0, 0, 0, 1], dtype=torch.bool)
    ans_top3: Final = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)
    ans_top5: Final = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)

    params__standard_inputs = {
        "top1": [(1,), [ans_top1]],
        "top3": [(3,), [ans_top3]],
        "top5": [(5,), [ans_top5]],
        "top1_5": [(1, 3, 5), [ans_top1, ans_top3, ans_top5]],
    }

    @pytest.mark.parametrize(
        "topk, expects",
        params__standard_inputs.values(),
        ids=params__standard_inputs.keys(),
    )
    def test__standard_inputs(self, sample_output, sample_target, topk, expects):
        results = fourier_attack.util.get_corrects(sample_output, sample_target, topk)
        for result, expect in zip(results, expects):
            assert result.equal(expect)
