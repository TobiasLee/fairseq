"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""
import torch

from fairseq.data import iterators
from fairseq.progress_bar import build_progress_bar
from plot.power_iter import Operator, deflated_power_iteration
# from hessian_eigenthings.lanczos import lanczos


class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
            self,
            args,
            task,
            model,
            epoch_itr,
            criterion,
            optimizer=None, 
            use_gpu=True,
            full_dataset=True
    ):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HVPOperator, self).__init__(size)
        self.args = args
        self.grad_vec = torch.zeros(size)
        self.model = model
        self.optimizer = optimizer 
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.epoch_itr = epoch_itr
        # Make a copy since we will go over it a bunch
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.task = task
        self.full_dataset=True

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec):
        # compute original gradient, tracking computation graph
        self.zero_grad()
        grad_vec = self.prepare_grad()
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
        )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        return hessian_vec_prod

    def _apply_full(self, vec):
        n = len(self.epoch_itr)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        self.criterion.train()
        self.model.train()
        grad_vec = None
        num_chunks = 0

        itr = self.epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=self.args.fix_batches_to_gpus,
            shuffle=(self.epoch_itr.epoch >= self.args.curriculum),
        )
        update_freq = (
            self.args.update_freq[self.epoch_itr.epoch - 1]
            if self.epoch_itr.epoch <= len(self.args.update_freq)
            else self.args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = build_progress_bar(
            self.args, itr, self.epoch_itr.epoch, no_progress_bar='simple',
        )

        for i, samples in enumerate(progress, start=self.epoch_itr.iterations_in_epoch):
            for i, sample in enumerate(samples):
                if sample is None:
                    # when sample is None, run forward/backward on a dummy batch
                    # and ignore the resulting gradients
                    # sample = self._prepare_sample(self._dummy_batch)
                    ignore_grad = True
                    continue
                else:
                    ignore_grad = False

                loss, sample_size, logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer, ignore_grad
                )

                grad_dict = torch.autograd.grad(
                    loss, self.model.parameters(), create_graph=True
                )
                if grad_vec is not None:
                    grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
                else:
                    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
                num_chunks += 1
        print('num chunks:%d' % num_chunks)
        # grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec


def compute_hessian_eigenthings(
        args,
        task,
        model,
        epoch_itr,
        criterion,
        optimizer=None,
        num_eigenthings=10,
        mode="power_iter",
        use_gpu=True,
        **kwargs
):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk

    num_eigenthings : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.

    mode : str ['power_iter', 'lanczos']
        which backend to use to compute the top eigenvalues.
    use_gpu:
        if true, attempt to use cuda for all lin alg computatoins

    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(
        args,
        task,
        model,
        epoch_itr,
        criterion,
        optimizer=optimizer,
        use_gpu=use_gpu,
    )
    eigenvals, eigenvecs = None, None
    if mode == "power_iter":
        eigenvals, eigenvecs = deflated_power_iteration(
            hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
        )
    # elif mode == "lanczos":
    #     eigenvals, eigenvecs = lanczos(
    #         hvp_operator, num_eigenthings, use_gpu=use_gpu, **kwargs
    #     )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return eigenvals, eigenvecs
