# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler
import torch

from ..adam import FairseqAdamAdaWU
from fairseq.logging import metrics


@register_lr_scheduler('adaptive_warmup')
class AdaptiveWarmupScheduler(FairseqLRScheduler):
    def __init__(self, args, optimizer, model=None):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # initial learning rate
        self.lr, self.global_lr = args.lr[0], args.lr[0]
        self.optimizer.set_lr(self.lr)
        self.weight_indicators = {'lo': 'decoder.layers.0.fc2.weight',
                                  'hi': 'decoder.layers.%d.fc2.weight' % (args.decoder_layers - 1)}
        assert model is not None, "We need a model reference for getting param stats"
        self.model = model
        self.lo_param, self.hi_param = None, None
        for name, param in self.model.named_parameters():
            if self.weight_indicators['lo'] in name:
                self.lo_param = param
            if self.weight_indicators['hi'] in name:
                self.hi_param = param
            # hyper params for ada-warmup
        self.bound_lo = args.bound_lo
        self.bound_hi = args.bound_hi
        self.beta3 = args.beta3
        self.beta4 = args.beta4
        # initial scale factor
        self.scale_factor = 1.0
        self.ratio_exp_avg = 0.0
        # after wu steps, we change back to invert sqrt decay
        self.decay_factor = warmup_end_lr * args.warmup_updates ** 0.5

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
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

        # adaptive warmup learning rate
        layer_lo, layer_hi = 1, 1

        if self.lo_param.grad is not None:
            layer_lo = self.lo_param.grad.data.float().norm().item()
        if self.hi_param.grad is not None:
            layer_hi = self.hi_param.grad.data.float().norm().item()

            # else:
            #     layer_hi = param.grad.data.float().norm().item()

        current_ratio = layer_lo / layer_hi  # current ratio is a scalar
        # print(num_updates, current_ratio)
        metrics.log_scalar('current_ratio', current_ratio)  # log to metrics for checkout
        del layer_hi, layer_lo
        if num_updates == 1:
            self.scale_factor = 1  # first compute ratio
            self.ratio_exp_avg = self.beta3 * self.ratio_exp_avg + (1 - self.beta3) * current_ratio
        elif 1 < num_updates < self.args.warmup_updates + 1:
            decay_ratio = current_ratio * (1 - self.beta3 ** (num_updates - 1)) / (self.ratio_exp_avg + 1e-9)
            if decay_ratio > self.bound_hi or decay_ratio < self.bound_lo:
                self.scale_factor /= 2
            # update s_t for learning rate adjustment
            self.scale_factor = self.scale_factor * self.beta4 + (1 - self.beta4) * 1.0
            # update ratio avg
            self.ratio_exp_avg = self.beta3 * self.ratio_exp_avg + (1 - self.beta3) * current_ratio
            # print("scale factor: %.9f  ratio_exp_avg: %.9f" % (self.scale_factor, self.ratio_exp_avg))
        elif self.args.warmup_updates + 1 <= num_updates:  # finish adaptive warmup steps,  we do not do decay anymore
            self.scale_factor = self.scale_factor * self.beta4 + (1 - self.beta4) * 1.0  # back to 1 within 100 steps
            self.lr = self.decay_factor * num_updates ** -0.5
            # self.optimizer.set_lr(self.lr * self.scale_factor)
            # return self.lr * self.scale_factor
        self.optimizer.set_lr(self.scale_factor * self.lr)
        return self.scale_factor * self.lr


@register_lr_scheduler('adaptive_warmup_term')
class AdaptiveWarmupScheduler(FairseqLRScheduler):
    def __init__(self, args, optimizer, model=None):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # initial learning rate
        self.lr, self.global_lr = args.lr[0], args.lr[0]
        self.optimizer.set_lr(self.lr)
        assert isinstance(self.optimizer, FairseqAdamAdaWU), 'we need a override adam for loss backward'
        assert model is not None, "We need a model reference for getting param stats"
        self.model = model
        self.lo_param, self.hi_param = None, None

        # hyper params for ada-warmup
        self.bound_lo = args.bound_lo
        self.bound_hi = args.bound_hi
        self.beta3 = args.beta3
        self.beta4 = args.beta4
        # initial scale factor
        self.scale_factor = 1.0
        self.ratio_exp_avg = 0.0
        # after wu steps, we change back to invert sqrt decay
        self.decay_factor = warmup_end_lr * args.warmup_updates ** 0.5

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
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

    def step_update(self, num_updates, loss=None):
        """Update the learning rate after each update."""
        if loss is None:
            self.optimizer.set_lr(self.scale_factor * self.lr)
            return self.scale_factor * self.lr

        dec_x0, dec_xL = self.model.get_decoder_states()
        grad_x0 = torch.grad(loss, dec_x0, retain_graph=True)
        grad_xL = torch.grad(loss, dec_xL, retain_graph=False)  # release graph

        layer_lo, layer_hi = 1, 1
        if grad_x0 is not None:
            layer_lo = grad_x0.data.float().norm().item()
        if grad_xL is not None:
            layer_hi = grad_xL.data.float().norm().item()
        current_ratio = layer_lo / layer_hi  # current ratio is a scalar
        metrics.log_scalar('current_ratio', current_ratio)  # log to metrics for checkout
        # print(num_updates, current_ratio)
        del layer_hi, layer_lo
        if num_updates == 1:
            self.scale_factor = 1  # first compute ratio
            self.ratio_exp_avg = self.beta3 * self.ratio_exp_avg + (1 - self.beta3) * current_ratio
        elif 1 < num_updates < self.args.warmup_updates + 1:
            decay_ratio = current_ratio * (1 - self.beta3 ** (num_updates - 1)) / (self.ratio_exp_avg + 1e-9)
            if decay_ratio > self.bound_hi or decay_ratio < self.bound_lo:
                self.scale_factor /= 2
            # update s_t for learning rate adjustment
            self.scale_factor = self.scale_factor * self.beta4 + (1 - self.beta4) * 1.0
            # update ratio avg
            self.ratio_exp_avg = self.beta3 * self.ratio_exp_avg + (1 - self.beta3) * current_ratio
            # print("scale factor: %.9f  ratio_exp_avg: %.9f" % (self.scale_factor, self.ratio_exp_avg))
        elif self.args.warmup_updates + 1 <= num_updates:  # finish adaptive warmup steps,  we do not do decay anymore
            self.scale_factor = self.scale_factor * self.beta4 + (1 - self.beta4) * 1.0  # back to 1 within 100 steps
            self.lr = self.decay_factor * num_updates ** -0.5
            # self.optimizer.set_lr(self.lr * self.scale_factor)
            # return self.lr * self.scale_factor
        self.optimizer.set_lr(self.scale_factor * self.lr)
        return self.scale_factor * self.lr
