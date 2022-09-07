from torch.optim.lr_scheduler import _LRScheduler
from typing import Union
import torch


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.7)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step


class TwoStepLR(_LRScheduler):

    def __init__(self, optimizer, lr, steps,  last_epoch=-1, verbose=False):
        self.learning_rate = lr
        self.steps = steps
        self.min_learning_rate: float = 5e-6
        self.passed = -1
        super(TwoStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.passed < self.steps:
            self.passed += 1
            return [(self.passed + 1.) / self.steps * group['lr'] for group in self.optimizer.param_groups]
        elif self.steps <= self.passed < self.steps * 2:
            self.passed += 1
            return [(2 - (self.passed + 1.) / self.steps) * (group['lr'] - self.min_learning_rate)+self.min_learning_rate for group in self.optimizer.param_groups]
        else:
            return [group['lr']*-0.1 for group in self.optimizer.param_groups]
