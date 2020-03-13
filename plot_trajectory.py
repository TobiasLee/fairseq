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

from plot import net_plotter
import plot.projection as proj
from plot import plot_2D
from train import distributed_main


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

    # Print args
    print(args)
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    # model = model.cuda() -> cause segmentation fault ? 
    criterion = task.build_criterion(args)
    print(model)
    #print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    #print('bp2')
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    extra_state = trainer.load_checkpoint( # we are loading an initialzied model, since the restore_file is fake
        args.restore_file,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )
    # --------------------------------------------------------------------------
    # load the init  model
    # --------------------------------------------------------------------------
    print('geting model...')
    w = net_plotter.get_weights(model)
    s = model.state_dict()
    print('finished')

    # --------------------------------------------------------------------------
    # collect models to be projected
    # --------------------------------------------------------------------------
    model_files = []
    print(args.start_epoch, args.end_epoch, args.save_epoch)
    for epoch in range(args.start_epoch, args.end_epoch + args.save_epoch, args.save_epoch):
        model_file = args.model_folder + '/' + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)

    # --------------------------------------------------------------------------
    # load or create projection directions
    # --------------------------------------------------------------------------
    # dir_file = net_plotter.name_direction_file(args)  # name the direction file

    if args.dir_file:
        dir_file = args.dir_file
    else:
        print('setting pca directions...')
        dir_file = proj.setup_PCA_directions(args, model_files, w, s, task=task)
    print('dir file:', dir_file)
    assert os.path.exists(dir_file), "direction file not exsits"
    # --------------------------------------------------------------------------
    # projection trajectory to given directions
    # --------------------------------------------------------------------------
    print('start plotting...')

    proj_file = proj.project_trajectory_fairseq(dir_file, w, s, model_files, args, task,
                                                args.dir_type, proj_method='cos')
    # proj_file = proj.project_trace(dir_file, w, model_files, args, task,
    #                                              proj_method='mean_dx')
    plot_2D.plot_trajectory(proj_file, dir_file)
    print('plotting finished')


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    # parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # model parameters
    parser.add_argument('--model-folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model-file', default='', help='path to the trained model file')
    parser.add_argument('--model-file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model-file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--init-model', action='store_true', default=False, help='x direction is trained model - initial model')

# direction parameters
    parser.add_argument('--dir-file', default='',
                        help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir-type', default='weights',
                        help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False,
                        help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf-file', default='',
                        help='customize the name of surface file, could be an existing file.')
    # checkpoint params
    parser.add_argument('--prefix', type=str, default='checkpoint', help='prefix of ckpt')
    parser.add_argument('--suffix', type=str, default='.pt', help='subfix of ckpt')
    parser.add_argument('--start-epoch', default=1, type=int, help='min index of epochs')
    parser.add_argument('--end-epoch', default=40, type=int, help='max number of epochs')
    parser.add_argument('--save-epoch', default=1, type=int, help='save models every few epochs')

    # plot parameters
    parser.add_argument('--proj-file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss-max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
