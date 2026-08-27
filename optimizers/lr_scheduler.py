# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import List

from torch import nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

__all__ = ["LinearLR", "ExponentialLR"]


class _LRSchedulerMONAI(_LRScheduler):
    """Base class for increasing the learning rate between two boundaries over a number
    of iterations"""

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            end_lr: the final learning rate.
            num_iter: the number of iterations over which the test occurs.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LRSchedulerMONAI, self).__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, t_total: int, cycles: float = 0.5, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
    
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from typing import Optional

# def make_warmup_cosine_then_cyclic(
#     optimizer,
#     warmup_epochs: int,
#     main_epochs: int,      # e.g., 300
#     total_epochs: int,     # args.max_epochs
#     eta_min: float = 1e-5, # LR floor for both phases
#     t0: Optional[int] = None, # first cycle length after main phase
#     t_mult: int = 1,       # growth factor for cycles
# ):
#     """
#     Phase 1: Linear warmup + cosine decay up to `main_epochs` (uses your MONAI scheduler).
#     Phase 2: CosineAnnealingWarmRestarts for the remaining epochs (epoch-cyclic).
#     """
#     # Phase 1 over [0, main_epochs)
#     phase1 = LinearWarmupCosineAnnealingLR(
#         optimizer=optimizer,
#         warmup_epochs=warmup_epochs,
#         max_epochs=main_epochs,     # IMPORTANT: only schedule until main_epochs
#         warmup_start_lr=0.0,
#         eta_min=eta_min,
#     )

#     remain = max(0, total_epochs - main_epochs)
#     if remain <= 0:
#         return phase1

#     # Phase 2 over [main_epochs, total_epochs)
#     # spread the remaining epochs over cycles; default: 2 equal cycles
#     if t0 is None:
#         t0 = max(1, remain // 2)
#     cawr = CosineAnnealingWarmRestarts(
#         optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
#     )

#     # Chain: phase1 → CAWR (step once per epoch in your loop, as usual)
#     return SequentialLR(
#         optimizer,
#         schedulers=[phase1, cawr],
#         milestones=[main_epochs],
#     )

def make_warmup_cosine_then_cyclic(
    optimizer,
    warmup_epochs: int,
    main_epochs: int,
    total_epochs: int,
    eta_min: float = 3e-5,
    t0: int = 50,
    t_mult: int = 1,
    post_cycle_max_lr: float = 1.5e-4,   # <— cap the peak in phase 2
    base_lr_for_ref: float = 3e-4,       # your original LR (optional, for logging)
):
    phase1 = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=warmup_epochs, max_epochs=main_epochs, eta_min=eta_min
    )

    remain = max(0, total_epochs - main_epochs)
    if remain <= 0:
        return phase1

    # full cycles that fit + any leftover
    num_full = remain // t0
    leftover = remain % t0

    scheds, milestones = [phase1], [main_epochs]

    # IMPORTANT: set the new base LR for phase 2 *before* constructing the next scheduler
    for g in optimizer.param_groups:
        g["lr"] = post_cycle_max_lr
        g["initial_lr"] = post_cycle_max_lr  # so PyTorch picks this as base_lrs

    if num_full > 0:
        cawr = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min)
        scheds.append(cawr)
        milestones.append(main_epochs + num_full * t0)

    if leftover > 0:
        # finish with plain cosine to decay into eta_min (no final jump)
        tail = CosineAnnealingLR(optimizer, T_max=leftover, eta_min=eta_min)
        scheds.append(tail)
        milestones.append(total_epochs)

    return SequentialLR(optimizer, schedulers=scheds, milestones=milestones[:-1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torch.optim import AdamW
    
    # Dummy model and optimizer
    model = torch.nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=3e-4)

    
    max_epochs = 600
    warmup_epochs = 50
    
    scheduler = make_warmup_cosine_then_cyclic(
        optimizer, warmup_epochs=30,
        main_epochs=300, total_epochs=600,
        eta_min=5e-5,
        t0=100,
        t_mult=1,
        post_cycle_max_lr=2e-4  # cap the restart amplitude
    )
    
    # Record LR per epoch
    lrs = []
    for epoch in range(max_epochs):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(range(max_epochs), lrs, label="LR schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning rate vs. Epochs")
    plt.grid(True)
    plt.legend()
    plt.show()