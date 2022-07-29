# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from asyncio import FastChildWatcher
import warnings

from sympy import arg

warnings.filterwarnings("ignore")
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from model.ftcn import I3D8x8
from engine import train_one_epoch, evaluate
import model.optimizer as optim
import utils
from data.dataset_clips import CelebDFClips, ForensicsClips_new32
from data.samplers import ConsecutiveClipSampler
from torchvision.transforms import Compose, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32_resume', action='store_true', default=False)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--frames', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--tt_use', default=True)
    # Model parameters
    parser.add_argument('--model', default='ftcn', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.5, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.01, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')  # ./pvt_v2_b5.pth

    # Dataset parameters
    parser.add_argument('--data-path', default='./', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    cudnn.benchmark = True

    train_transform = Compose(
        [CenterCrop(224), RandomHorizontalFlip(), ToTensor(),
         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    test_transform = Compose(
        [CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    train_split = pd.read_json('/data-x/g15/ffpp-faces/train.json', dtype=False)
    train_files_real, train_files_fake = utils.get_files_from_split(train_split)

    # gpu3 bs=2 --> train 15 ; test 25
    # 270 for train and 110 for test, as recommended in FF++
    real_data = ForensicsClips_new32(train_files_real,
                                     train_files_fake,
                                     args.frames,
                                     grayscale=False,
                                     compression='c23',
                                     ds_types=["Origin"],
                                     transform=train_transform,
                                     max_frames_per_video=320,
                                     )
    fake_data = ForensicsClips_new32(train_files_real,
                                     train_files_fake,
                                     args.frames,
                                     grayscale=False,
                                     compression='c23',
                                     ds_types=['Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'],
                                     transform=train_transform,
                                     max_frames_per_video=320,
                                     )

    train_dataset = torch.utils.data.ConcatDataset([real_data, fake_data])

    # 测试集frames可以尝试16
    validate_dataset = CelebDFClips(args.frames, False, test_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = ConsecutiveClipSampler(validate_dataset.clips_per_video)
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                    num_workers=args.num_workers, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size * 2,
                                                  sampler=test_sampler, num_workers=args.num_workers)
    print(f"=====>Totally {len(train_dataset)} train video clips...")
    print(f"=====>Totally {len(validate_dataset)} test videos...")
    print(sum(validate_dataset.clips_per_video))

    # 考虑帧间连续性， 去掉mix_up??      --效果不佳
    # 过拟合仍然是重要问题
    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    mixup_active = args.mixup > 0

    print(f"Creating model: {args.model}")
    model = I3D8x8()
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)

    model_ema = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    opt_method = "sgd"
    optimizer = optim.construct_optimizer(model, opt_method)
    threshold = 0.5
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()
    output_dir = Path(args.output_dir)

    if args.eval:
        auc, test_stats = evaluate(data_loader_val, model, threshold, device)
        print(f"Auc of the network on the {len(validate_dataset)} test videos: {auc}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, threshold,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / str('checkpoint_{}.pth'.format(epoch))]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        auc, test_stats = evaluate(data_loader_val, model, threshold, device)
        print(f"AUC of the network on the {len(validate_dataset)} test videos: {auc}")
        max_auc = max(max_auc, auc)
        print(f'Max AUC: {max_auc}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'Test_AUC': auc}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
