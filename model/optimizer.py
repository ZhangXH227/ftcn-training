# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

# import utils.lr_policy as lr_policy


def construct_optimizer(model, opt_method):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": 0.0},
        {"params": non_bn_parameters, "weight_decay": 1e-4},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    if opt_method == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            dampening=0.0,
            nesterov=True,
        )
    # elif opt_method == "adam":
    #     return torch.optim.Adam(
    #         optim_params,
    #         lr=cfg.SOLVER.BASE_LR,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #     )
    # elif opt_method == "adamw":
    #     return torch.optim.AdamW(
    #         optim_params,
    #         lr=cfg.SOLVER.BASE_LR,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    #     )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(opt_method)
        )


# def get_epoch_lr(cur_epoch, cfg):
#     """
#     Retrieves the lr for the given epoch (as specified by the lr policy).
#     Args:
#         cfg (config): configs of hyper-parameters of ADAM, includes base
#         learning rate, betas, and weight decays.
#         cur_epoch (float): the number of epoch of the current training stage.
#     """
#     return lr_policy.get_lr_at_epoch(cfg, cur_epoch)
#
#
# def set_lr(optimizer, new_lr):
#     """
#     Sets the optimizer lr to the specified value.
#     Args:
#         optimizer (optim): the optimizer using to optimize the current network.
#         new_lr (float): the new learning rate to set.
#     """
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = new_lr
