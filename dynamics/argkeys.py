from dynamics.data.simulated.gaussian_mixture import GaussianMixture
from dynamics.data.simulated.random_mixture import RandomMixture
from dynamics.models.linear_network import LinearNetwork
import torch.optim as opt
import torch.nn as nn
import argparse
import json
import numpy as np

OPTIMIZERS = {
    "adam": opt.Adam,
    "sgd": opt.SGD
}

MODELS = {
    "linear_network": LinearNetwork
}

DATASETS = {
    "gaussian_mixture": GaussianMixture,
    "random_mixture": RandomMixture
}

CRITERION = {
    "mse": nn.MSELoss,
    "ce": nn.CrossEntropyLoss
}

ARGKEYS = {
    "optimizer": OPTIMIZERS,
    "model": MODELS,
    "dataset": DATASETS,
    "criterion": CRITERION
}


def create_parser(args):
    parser = argparse.ArgumentParser("Dynamics")
    for k, v in args.items():
        parser.add_argument("--%s" % k, default=v, type=type(v))
    return parser


def resolve_args(args):
    for k, v in args.__dict__.items():
        kk = ARGKEYS.get(k, None)
        if kk is not None:
            setattr(args, k, kk[v])
        elif "_kwargs" in k or "_args" in k:
            setattr(args, k, eval(v))
    return args
