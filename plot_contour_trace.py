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
from plot.plot_surface import setup_surface_file, name_surface_file
from trans_plot import plot as plot_surface


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
    # print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    # print('bp2')
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    if args.init_model:
        print('initialize model with a random one')
        args.restore_file = 'random_ckpt.pt'
    _ = trainer.load_checkpoint(  # use this code to restore ckpt
        args.restore_file,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        eval(args.optimizer_overrides),
        reset_meters=args.reset_meters,
    )
    #
    print('loading model...')
    w = net_plotter.get_weights(model)
    print('finished')

    # collect models to be projected
    model_files = []
    print(args.start_epoch, args.end_epoch, args.save_epoch)
    for epoch in range(args.start_epoch, args.end_epoch + args.save_epoch, args.save_epoch):
        model_file = args.model_folder + '/' + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)

    # load or create projection directions PCA + normalize
    if args.dir_file:
        dir_file = args.dir_file
        assert os.path.exists(dir_file), "direction file not exsits"
    else:
        print('setting pca directions...')
        # theta_1 - theta_n , theta_2 - theta_n , ... , theta_{n-1} - theta_n
        dir_file = proj.setup_PCA_directions_normalize(args, model_files, w, task=task)
    print('dir file:', dir_file)

    # plotting loss surface
    surf_file = name_surface_file(args, dir_file)
    surf_exist = setup_surface_file(args, surf_file, dir_file)
    direction = net_plotter.load_directions(dir_file)

    if not surf_exist:  # if the surface already exists
        # mpi setting, we do not use mpi to avoid bug
        comm, rank, nproc = None, 0, 1
        # dataset and iteraotr
        dataset2plot = args.valid_subset.split(',')[0]  # we use valid_subset as the dataset to plot on
        epoch_itr = trainer.get_train_iterator(epoch=0)
        # weights and state
        weight = net_plotter.get_weights(model)  # initial parameters
        state = copy.deepcopy(model.state_dict())  # deepcopy since state_dict are references
        plot_surface(surf_file, model, weight, state, direction,
                     comm, rank, args,
                     trainer, task, epoch_itr, dataset2plot,
                     loss_key='loss')

    # --------------------------------------------------------------------------
    # projection trajectory to given directions
    # --------------------------------------------------------------------------
    print('start plotting trajectory')
    proj_file = proj.project_trajectory_fairseq(dir_file, w, model_files, args, task,
                                                args.dir_type, proj_method='lstsq')
    plot_2D.plot_contour_trajectory(surf_file, dir_file, proj_file,
                                    surf_name='loss', ckpt='init' if args.init_model else 'best')
    print('plotting finished')


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use for each rank, useful for data parallel evaluation')
    # model parameters
    parser.add_argument('--model-folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--init-model', action='store_true', default=False,
                        help='theta_n treat start point as a initialization file')

    # direction parameters
    parser.add_argument('--dir-file', default='',
                        help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir-folder', default='',
                        help='folder for store the direction file')

    parser.add_argument('--dir-type', default='weights',
                        help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
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
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
