import torch
import torch.nn as nn

AVAILABLE_OPTIMIZERS = ["sgd", "adam", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None, lr_multiplier=1.0):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
        lr_multiplier (float): multiplier for the learning rate (for domain-specific LR)
    """
    if optim_cfg.NAME not in AVAILABLE_OPTIMIZERS:
        raise ValueError(
            "Optimizer must be one of {}, but got {}".format(
                AVAILABLE_OPTIMIZERS, optim_cfg.NAME
            )
        )

    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model

    # Apply learning rate multiplier
    effective_lr = optim_cfg.LR * lr_multiplier
    
    if optim_cfg.NAME == "sgd":
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=effective_lr,
            momentum=optim_cfg.MOMENTUM,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            dampening=optim_cfg.SGD_DAMPENING,
            nesterov=optim_cfg.SGD_NESTEROV,
            )
    elif optim_cfg.NAME == "adam":
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr=effective_lr,
            weight_decay=optim_cfg.WEIGHT_DECAY
        )
    elif optim_cfg.NAME == "adamw":
        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=effective_lr,
            weight_decay=optim_cfg.WEIGHT_DECAY
        )
    else:
        raise ValueError("Unknown optimizer: {}".format(optim_cfg.NAME))
    return optimizer
