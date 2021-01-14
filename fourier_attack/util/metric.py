from typing import Final, Iterable, List

import torch


def get_corrects(
    output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)
) -> List[torch.Tensor]:
    """Get list of bool tensor which represents the estimation's correctness.

    This function returns list of bool tensor.
    Each bool tensor represents the estimation's correctness under topk condition.

    Parameters
    ----------
    output : torch.Tensor
        The output tensor from model.
    target : torch.Tensor
        The size of input image which is represented by 2D tensor.
    topk : Iterable
        The Iterable of int which you want to get corrects.
    """
    with torch.no_grad():
        batch_size: Final = output.size(0)

        # return the k larget elements. top-k index: size (b, k).
        _, pred = output.topk(max(topk), dim=1)

        corrects = pred.eq(target[:, None].expand_as(pred))

        return_list = []
        for k in topk:
            corrects_ = corrects[:, :k].view(
                batch_size, -1
            )  # view is needed when topk=(1,)
            return_list.append(corrects_.sum(dim=-1, dtype=torch.bool))

    return return_list
