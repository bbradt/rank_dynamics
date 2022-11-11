import numpy as np
import torch
from inspect import currentframe, getframeinfo
import os
import warnings


def make_covariance(method, dim, *args, **kwargs):
    if method == "zeros":
        A = np.zeros((dim, dim))
    elif method == "ones":
        A = np.ones((dim, dim))
    elif method == "eye":
        A = np.eye(dim)
    elif method in ["rand", "randn"]:
        A = getattr(np.random, method)(dim, dim)
    elif method in ["ranf"]:
        A = getattr(np.random, method)((dim, dim))
    else:
        A = getattr(np.random, method)(*args, size=(dim, dim), **kwargs)
    C = A.T @ A
    C = C / np.max(C)
    np.fill_diagonal(C, 1)
    return C


def make_means(method, dim, *args, **kwargs):
    if method == "zeros":
        A = np.zeros((dim,))
    elif method == "ones":
        A = np.ones((dim,))
    elif method in ["rand", "randn"]:
        A = getattr(np.random, method)(dim)
    elif method in ["ranf"]:
        A = getattr(np.random, method)((dim,))
    else:
        A = getattr(np.random, method)(*args, size=(dim,), **kwargs)
    return A.flatten()


ALLOWED_DISTRIBUTIONS = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Dirichlet",
    "Exponential",
    "FisherSnedecor",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Kumaraswamy",
    # "LKJCholesky", # only returns covariance
    "Laplace",
    "LogNormal",
    # "LowRankMultivariateNormal", # Currently not working for some reason
    "Multinomial",
    "MultivariateNormal",
    # "NegativeBinomial",
    # "Normal", # 1 Dimensional - pointless
    "Pareto",
    "Poisson",
    # "RelaxedBernoulli",
    "StudentT",
    "Uniform",
    "VonMises",
    "Weibull",
    "Wishart"
]

DEFAULT_DIST_KWARGS = {
    "Bernoulli": [0.3],
    "Beta": [0.5, 0.5],
    "Binomial": [20, 0.5],
    "Cauchy": [0.0, 1.0],
    "Chi2": [1.0],
    "ContinuousBernoulli": [0.3],
    "Dirichlet": [0.5],
    "Exponential": [1.0],
    "FisherSnedecor": [1.0, 2.0],
    "Gamma": [1.0, 1.0],
    "Geometric": [0.3],
    "Gumbel": [1.0, 2.0],
    "HalfCauchy": [1.0],
    "HalfNormal": [1.0],
    "Kumaraswamy": [1.0, 1.0],
    # "LKJCholesky": [0.5],
    "Laplace": [0.0, 1.0],
    "LogNormal": [0.0, 1.0],
    # "LowRankMultivariateNormal": [],
    # "Multinomial": [100, 0.5],
    "MultivariateNormal": [],
    # "NegativeBinomial": [],
    # "Normal": [0, 1],
    "Pareto": [1.0, 1.0],
    "Poisson": [4],
    # "RelaxedBernoulli",
    "StudentT": [2.0],
    "Uniform": [0, 1],
    # "VonMises": [1, 1],
    "Weibull": [1, 1],
    "Wishart": [2]
}


def make_serializable(dictj):
    def copy_and_modify(d):
        if isinstance(d, dict):
            d2 = {k: copy_and_modify(v) for k, v in d.items()}
            return d2
        elif type(d) is type:
            return str(d)
        return d
    return copy_and_modify(dictj)


def resolve_list_of(datatype, length, args, none_allowed=True, uniform=True, gen=None, gen_args=[]):
    if gen is not None:
        args = make_means(gen, length, *gen_args)
    elif type(args) is datatype or (args is None and none_allowed):
        args = [args for _ in range(length)]
    elif type(args) is list:
        assert len(
            args) == length, "Length of N must be same as number of modes"
        if uniform:
            for elem in args:
                if type(elem) is not datatype:
                    raise (ValueError("Expected a uniform list of %s but found an element of type %s" % (
                        datatype, type(elem))))
        if len(args) == 0 and not none_allowed:
            raise ValueError(
                "this input cannot be empty list")
    elif not none_allowed:
        raise ValueError(
            "this input cannot be None, must be %s or list" % str(datatype))
    return args


def get_distribution_with_dim(distribution_name, dim, args=None,
                              loc_method="zeros",
                              loc_args=[],
                              cov_method="eye",
                              cov_args=[],
                              gen_method=None,
                              gen_args=None):
    if distribution_name not in ALLOWED_DISTRIBUTIONS:
        raise (ValueError("Distribution %s is not supported" % distribution_name))
    if args is None:
        if distribution_name in DEFAULT_DIST_KWARGS.keys():
            args = DEFAULT_DIST_KWARGS[distribution_name]
        else:
            raise (ValueError("Provide some args for distribution %s" %
                   distribution_name))
    distr = getattr(torch.distributions, distribution_name)
    if distribution_name in ["LowRankMultivariateNormal", "MultivariateNormal", "Wishart"]:
        if distribution_name == "LowRankMultivariateNormal":
            loc = make_means(loc_method, dim, *loc_args)
            cov_factor = make_covariance(cov_method, dim, *cov_args)
            cov_diag = make_covariance(cov_method, dim, *cov_args)
            args = [loc, cov_factor, cov_diag]
        elif distribution_name == "MultivariateNormal":
            loc = make_means(loc_method, dim, *loc_args)
            cov = make_covariance(cov_method, dim, *cov_args)
            args = [loc, cov]
        elif distribution_name == "Wishart":
            assert (
                type(args[0]) is not list
            ), "Args[0] should be numeric for Wishart"
            df = make_covariance(cov_method, dim, *cov_args)
            args = [df, [args[0]]]
    else:
        if gen_method is not None:
            new_args = []
            for ai, arg in enumerate(args):
                newarg = resolve_list_of(
                    type(arg), dim, arg, gen=gen_method, gen_args=gen_args)
                constraint = list(distr.arg_constraints.values())[ai]
                if constraint.check(torch.Tensor(newarg)).all():
                    new_args.append(newarg)
                else:
                    new_args.append([arg for _ in range(dim)])
            args = new_args
    new_args = None
    if len(args) > 0 and type(args[0]) not in [list, np.ndarray]:
        new_args = []
        for arg in args:
            newarg = torch.Tensor([arg for _ in range(dim)])
            new_args.append(newarg)
    elif len(args) == 0:
        new_args = torch.Tensor([args for args in range(dim)])
    else:
        if len(args[0]) != dim:
            warnings.warn("Dimension will be set to %s instead of %s" %
                          (len(args[0]), dim), Warning)
        new_args = [torch.Tensor(arg) for arg in args]
    distr = distr(*new_args)
    return distr


def dprint(*args):
    frameinfo = getframeinfo(currentframe().f_back)
    print(os.path.basename(frameinfo.filename),
          currentframe().f_back.f_lineno, *args)


def chunks(l, n):
    """Split an array l into n chunks, using pigeon-hole principle
        Args: 
            l: list<Object> - a list of objects to split
            n: int - the number of chunks to split

    """
    return np.array_split(np.array(l), n)


def mm_flatten(*tensors):
    """A wrapper for flattening tensors in a list so that they may be 
        multiplied using torch.mm

        Args:
            *tensors - list of pytorch tensors to be multiplied
    """
    if len(tensors[0].shape) > 2:
        dims = list(range(len(tensors[0].shape)))
        return [t.flatten(*dims[:-1]) for t in tensors]
    return tensors


def point_send(source, dst, tensor, device, sub_group):
    #print("Coordinating shape ", source, dst)
    tensor = coordinate_shape(tensor, device, source, sub_group)
    #print("Sending from, to", source, dst)
    torch.distributed.barrier(sub_group)
    torch.distributed.broadcast(tensor=tensor, src=source, group=sub_group)
    torch.distributed.barrier(sub_group)
    return tensor


def coordinate_shape(tensor, device, src, group, ndim=2):
    #print("Tensor is ", tensor)
    shape = torch.zeros(ndim).to(device)
    if tensor is not None:
        shape = torch.Tensor(list(tensor.shape)).to(device)
    #print("shape b4", shape)
    torch.distributed.broadcast(tensor=shape, src=src, group=group)
    torch.distributed.barrier(group)
    #print("shape aft", shape)
    if tensor is None:
        shape = [int(s) for s in shape.tolist()]
        return torch.zeros(*shape).to(device)
    else:
        return tensor


def point_to_master(tensor, world_size, device, master_rank=0):
    rank = torch.distributed.get_rank()
    #print("We are on rank", rank)
    recv = []
    for i in range(world_size):
        if i == master_rank:
            continue
        sub_group = torch.distributed.new_group([i, master_rank])
        torch.distributed.barrier(sub_group)
        if rank != master_rank:
            recv.append(point_send(i, master_rank, tensor, device, sub_group))
            #print("Sent Size", recv[-1].size())
        else:
            recv.append(point_send(i, master_rank, tensor, device, sub_group))
            #print("Received Size", recv[-1].size())
        torch.distributed.destroy_process_group(sub_group)

    return recv


def coordinated_broadcast(tensor, device, src, group=None):
    tensor = coordinate_shape(tensor, device, src, group)
    torch.distributed.broadcast(tensor=tensor, src=src, group=group)
    torch.distributed.barrier(group)
    return tensor
