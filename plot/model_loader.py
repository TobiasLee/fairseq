# import os
# import cifar10.model_loader
from fairseq import checkpoint_utils
from fairseq.models.transformer import TransformerModel


# def load(dataset, model_name, model_file, data_parallel=False):
#     if dataset == 'cifar10':
#         net = cifar10.model_loader.load(model_name, model_file, data_parallel)
#     return net


def load_transformer(args, task, model_file):
    net = TransformerModel.build_model(args, task)
    state = checkpoint_utils.load_checkpoint_to_cpu(model_file)
    try:
        net.load_state_dict(model_file, state["model"], strict=True, args=args)
    except Exception:
        raise Exception(
            "Cannot load model parameters from checkpoint {}; "
            "please ensure that the architectures match.".format(model_file)
        )
    return net
