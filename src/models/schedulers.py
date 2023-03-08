from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupToConstant(_LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value."""

    def __init__(self, optimizer: Optimizer, num_steps: int = 100) -> None:
        """
        args:
            optimizer (Optimizer): Wrapped optimizer.
            num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self.last_epoch > self.num_steps:
            return [base_lr for base_lr in self.base_lrs]
        return [
            (base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs
        ]
