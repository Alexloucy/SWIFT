from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR

AVAILABLE_LR_SCHEDULERS = ["Cosine", "StepLR"]


class _BaseWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_epoch, last_epoch=-1):
        self.scheduler = scheduler
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.scheduler.step(epoch)
            self._last_lr = self.scheduler.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    def __init__(self, optimizer, scheduler, warmup_epoch, cons_lr, last_epoch=-1):
        self.cons_lr = cons_lr
        super().__init__(optimizer, scheduler, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.scheduler.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

class LinearWarmupScheduler(_BaseWarmupScheduler):
    def __init__(self, optimizer, scheduler, warmup_epoch, last_epoch=-1):
        super().__init__(optimizer, scheduler, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.scheduler.get_last_lr()
        return [base_lr * (self.last_epoch + 1) / self.warmup_epoch for base_lr in self.base_lrs]

def build_lr_scheduler(optimizer, optim_cfg, scheduler_type=None, step_size=None):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
        scheduler_type (str, optional): Override scheduler type for domain-specific config.
        step_size (int, optional): Override step size for StepLR scheduler.
    """
    
    # Use override parameters if provided (for domain-specific configs)
    lr_scheduler_type = scheduler_type if scheduler_type is not None else optim_cfg.LR_SCHEDULER
    step_size_value = step_size if step_size is not None else getattr(optim_cfg, 'STEP_SIZE', 5)

    if lr_scheduler_type not in AVAILABLE_LR_SCHEDULERS:
        raise ValueError(
            "LR Scheduler must be one of {}, but got {}".format(
                AVAILABLE_LR_SCHEDULERS, lr_scheduler_type
            )
        )

    if lr_scheduler_type == "Cosine":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=optim_cfg.MAX_EPOCH, eta_min=1e-6)
    elif lr_scheduler_type == "StepLR":
        scheduler = StepLR(optimizer=optimizer, step_size=step_size_value)

    if optim_cfg.WARMUP_TYPE == "constant":
        scheduler = ConstantWarmupScheduler(
            optimizer, scheduler, optim_cfg.WARMUP_EPOCH, optim_cfg.WARMUP_CONS_LR
        )
    elif optim_cfg.WARMUP_TYPE == "linear":
        scheduler = LinearWarmupScheduler(
            optimizer, scheduler, optim_cfg.WARMUP_EPOCH
        )

    return scheduler
