from typing import Union

import numpy as np
import torch as T
import torch.nn as nn


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    """Return a pytorch loss function given a name."""
    if name == "none":
        return None

    # Regression losses
    if name == "huber":
        return nn.HuberLoss(reduction="none")
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")


def to_np(inpt: Union[T.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()
