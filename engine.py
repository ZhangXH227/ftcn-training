# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import os
from operator import index
import sys
from typing import Iterable, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, threshold, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,fp32=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for param in model.parameters():
        param.requires_grad = True

    for samples, targets, video_indices in metric_logger.log_every(data_loader, print_freq, header):
        # print(samples.shape)
        # print("Labels:", targets, "   ||    video index:", video_indices)
        targets = targets.cuda(non_blocking=True)
        samples = samples.cuda(non_blocking=True)
        # print("Before Mixup:", samples, targets)   # [0,0,0,0]
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print("After Mixup:", samples, targets)    # [[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]]
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            targets = targets.to(torch.float32)
            preds = (outputs > 0.500).to(torch.float32)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, targets)
            acc = torch.sum(preds == targets.data).to(torch.float32)/len(targets.data)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)



        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        for param in model.parameters():
            param.requires_grad = True

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        # loss.backward()
        # optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from sklearn import metrics

def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])
    print(output_batch, output_labels, f"All {len(output_labels.cpu().numpy())} videos done!")

    fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())
    return metrics.auc(fpr, tpr)

from collections import defaultdict
import torch.nn.functional as F

@torch.no_grad()
def evaluate(data_loader, model, threshold, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    video_to_logits = defaultdict(list)
    video_to_labels = {}

    for images, target, video_indices in metric_logger.log_every(data_loader, 10, header):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            target = target.to(torch.float32)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, target)
            sigmoid = nn.Sigmoid()
            score = sigmoid(outputs)

        for i in range(len(video_indices)):
                video_id = video_indices[i].item()
                video_to_logits[video_id].append(score[i])
                video_to_labels[video_id] = target[i]

        if isinstance(images, (list,)):
            batch_size = images[0].shape[0]
        else:
            batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        
    auc_video = compute_video_level_auc(video_to_logits, video_to_labels)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('*loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    print("=====>Video-Level-AUC:", auc_video)

    return auc_video, {k: meter.global_avg for k, meter in metric_logger.meters.items()}
