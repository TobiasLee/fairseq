# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('adaptive_warmup')
class AdaptiveWarmupScheduler(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer, model=None):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        #if args.warmup_init_lr < 0:
        #    args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # initial learning rate
        self.lr = args.lr[0]
        self.optimizer.set_lr(self.lr)
        self.weight_indicators = {'lo': 'decoder.layers.0.fc2.weight',
                                  'hi': 'decoder.layers.%d.fc2.weight' % (args.decoder_layers - 1)}
        assert model is not None, "We need a model reference for getting param stats"
        self.model = model
        # hyper params for ada-warmup
        self.bound_lo = args.bound_lo
        self.bound_hi = args.bound_hi
        self.beta3 = args.beta3
        self.beta4 = args.beta4
        # initial scale factor
        self.scale_factor = 1.0
        self.ratio_exp_avg = 1.0

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        # parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
        #                     help='warmup the learning rate linearly for the first N updates')
        # parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
        #                     help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--bound_lo', default=0.75, type=float, metavar='BLO',
                            help='ratio lower bound')
        parser.add_argument('--bound_hi', default=1.5, type=float, metavar='BHI',
                            help='ratio higher bound')
        parser.add_argument('--beta3', default=0.99, type=float, metavar='BT3',
                            help='ratio higher bound')
        parser.add_argument('--beta4', default=0.995, type=float, metavar='BT4',
                            help='ratio higher bound')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        # initial setting
        if num_updates == 0:
            self.optimizer.set_lr(self.lr)
            return self.lr

        # adaptive warmup learning rate
        layer_lo, layer_hi = None, None
        for name, param in self.model.named_parameters():
            if self.weight_indicators['lo'] in name:
                if len(self.optimizer.optimizer.state[param]) ==0:
                    self.optimizer.set_lr(self.lr)
                    return self.lr
                layer_lo = self.optimizer.optimizer.state[param]['exp_avg'].data.float().norm()
            if self.weight_indicators['hi'] in name:
                layer_hi = self.optimizer.optimizer.state[param]['exp_avg'].data.float().norm()

        # assert layer_lo is not None
        current_ratio = layer_lo.item() / layer_hi.item()  # current ratio is a scalar
        if num_updates == 1:
            self.scale_factor = 1  # step 1
            self.ratio_exp_avg = self.beta3 * 0 + (1 - self.beta3) * current_ratio
        elif num_updates > 1:
            decay_ratio = current_ratio * (1 - self.beta3 ** num_updates) / self.ratio_exp_avg
            # print(decay_ratio)
            if decay_ratio > self.bound_hi or decay_ratio < self.bound_lo:
                self.scale_factor /= 2
            # update s_t for learning rate adjustment
            self.scale_factor = self.scale_factor * self.beta4 + (1 - self.beta4) * 1.0
            # update ratio avg
            self.ratio_exp_avg = self.beta3 * self.ratio_exp_avg + (1 - self.beta3) * current_ratio
        self.optimizer.set_lr(self.scale_factor * self.lr)
        return self.scale_factor * self.lr

