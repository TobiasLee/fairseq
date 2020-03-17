"""
    Plot the optimization path in the space spanned by principle directions.
"""
from random import random

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.models.transformer import TransformerModel
from hessian_compute import compute_hessian_eigenthings
from train import train


def main(args, init_distributed=False):
    utils.import_user_module(args)
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify  batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    task = tasks.setup_task(args)
    try:
        args.xmin, args.xmax, args.xnum = [int(a) for a in args.x.split(':')]
        print(args.xmin, args.xmax, args.xnum)
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [int(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
                'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    # model = model.cuda() -> cause segmentation fault ?
    criterion = task.build_criterion(args)
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    if args.init_model:
        print('project no warmup ckpt trace')
        # args.restore_file = 'random_ckpt.pt'
    _ = trainer.load_checkpoint(  # use this code to restore ckpt
        args.restore_file,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )
    epoch_itr = trainer.get_train_iterator(epoch=0)
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        args,
        task,
        model,
        epoch_itr,
        criterion,
        num_eigenthings=10,
        mode="power_iter",
        use_gpu=torch.cuda.is_available()
    )
    print(eigenvals)
    print(eigenvecs)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
