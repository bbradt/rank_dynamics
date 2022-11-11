from dynamics.utils import get_distribution_with_dim, make_means, resolve_list_of
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Union
import copy
import os


DEFAULT_MODES = 1
DEFAULT_N = 64
DEFAULT_ARGS = None
DEFAULT_KWARGS = None
DEFAULT_DISTRIBUTIONS = "MultivariateNormal"
DEFAULT_EMBEDDING_DIM = 2
DEFAULT_SOURCE_DIM = 2
DEFAULT_LOC_METHODS = "zeros"
DEFAULT_LOC_ARGS = None
DEFAULT_COV_METHODS = "eye"
DEFAULT_COV_ARGS = None


class RandomMixture(Dataset):
    def __init__(self,
                 modes: int = DEFAULT_MODES,
                 N: Union[int, list] = DEFAULT_N,
                 args: list = DEFAULT_ARGS,
                 kwargs: Union[dict, list] = DEFAULT_KWARGS,
                 distributions: Union[str, list] = DEFAULT_DISTRIBUTIONS,
                 source_dim: Union[int, list] = DEFAULT_SOURCE_DIM,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 loc_methods: Union[str, list] = DEFAULT_LOC_METHODS,
                 loc_args: list = DEFAULT_LOC_ARGS,
                 cov_methods: Union[str, list] = DEFAULT_COV_METHODS,
                 cov_args: list = DEFAULT_COV_ARGS,
                 gen_methods: Union[str, list] = None,
                 gen_args: list = None
                 ):
        super(RandomMixture, self).__init__()
        # Resolve number of samples per mode
        N = resolve_list_of(int, modes, N, none_allowed=False,
                            uniform=True, gen=None, gen_args=None)
        Nss = np.cumsum(N)
        # Resolve args per mode
        args = resolve_list_of(
            list, modes, args, none_allowed=True, uniform=True, gen=None, gen_args=None)
        # Resolve kwargs
        kwargs = resolve_list_of(
            dict, modes, kwargs, none_allowed=True, uniform=True, gen=None, gen_args=None)
        # Resolve distributions
        distributions = resolve_list_of(
            str, modes, distributions, none_allowed=False, uniform=True, gen=None, gen_args=None)
        # Resolve source dim
        source_dim = resolve_list_of(
            int, modes, source_dim, none_allowed=False, uniform=True, gen=None, gen_args=None)
        # Loc
        if loc_methods is not None:
            loc_methods = resolve_list_of(
                str, modes, loc_methods, none_allowed=False, uniform=True, gen=None, gen_args=None)
            loc_args = resolve_list_of(
                list, modes, loc_args, none_allowed=True, uniform=True, gen=None, gen_args=None
            )
        # Cov
        if cov_methods is not None:
            cov_methods = resolve_list_of(
                str, modes, cov_methods, none_allowed=False, uniform=True, gen=None, gen_args=None)
            cov_args = resolve_list_of(
                list, modes, cov_args, none_allowed=True, uniform=True, gen=None, gen_args=None
            )
        # Other Generators
        if gen_methods is not None:
            gen_methods = resolve_list_of(
                str, modes, gen_methods, none_allowed=True, uniform=True, gen=None, gen_args=None
            )
            gen_args = resolve_list_of(
                list, modes, gen_args, none_allowed=True, uniform=True, gen=None, gen_args=None
            )
        # Create Empty Mixture
        mixture = torch.zeros((max(Nss), embedding_dim))
        labs = torch.zeros((max(Nss),))
        # For each mode
        Nlast = 0
        for mi in range(modes):
            mode_N = Nss[mi]
            mode_args = args[mi]
            #mode_kwargs = kwargs[mi]
            mode_name = distributions[mi]
            mode_source_dim = source_dim[mi]
            mode_loc_method = mode_loc_args = None
            if loc_methods is not None:
                mode_loc_method = loc_methods[mi]
                mode_loc_args = loc_args[mi]
            mode_cov_method = mode_cov_args = None
            if cov_methods is not None:
                mode_cov_method = cov_methods[mi]
                mode_cov_args = cov_args[mi]
            mode_gen = mode_gen_args = None
            if gen_methods is not None:
                mode_gen = gen_methods[mi]
                mode_gen_args = gen_args[mi]
            distribution = get_distribution_with_dim(mode_name,
                                                     mode_source_dim,
                                                     mode_args,
                                                     loc_method=mode_loc_method,
                                                     loc_args=mode_loc_args,
                                                     cov_method=mode_cov_method,
                                                     cov_args=mode_cov_args,
                                                     gen_method=mode_gen,
                                                     gen_args=mode_gen_args)
            sample = distribution.sample((mode_N-Nlast, ))
            Uv, Sv, Vv = torch.linalg.svd(sample, full_matrices=False)
            Sv = torch.diag_embed(Sv)
            if mode_source_dim < embedding_dim:
                # project into higher dimensional space
                offset_min = 0
                offset_max = embedding_dim - Vv.shape[0] - 1
                if offset_max == offset_min or offset_max <= 0:
                    offset = offset_min
                else:
                    offset = max(np.random.randint(offset_min, offset_max), 0)
                #offset = 0
                Uu = torch.zeros((mode_N-Nlast, embedding_dim))
                Su = torch.zeros((embedding_dim, embedding_dim))
                Vu = torch.zeros((embedding_dim, embedding_dim))
                Uu[:, offset:(Uv.shape[1]+offset)] = Uv
                Vu[offset:(Vv.shape[0]+offset),
                   offset:(Vv.shape[1]+offset)] = Vv
                Su[offset:(Sv.shape[0]+offset),
                   offset:(Sv.shape[1]+offset)] = Sv
                sample = Uu @ Su @ Vu
            elif mode_source_dim > embedding_dim:
                Sz = Sv[:embedding_dim, :embedding_dim]
                sample = Uv[:,
                            :embedding_dim] @ Sz @ Vv[:embedding_dim, :embedding_dim]
            mixture[Nlast:mode_N, :] += sample
            labs[Nlast:mode_N] = mi
            Nlast = mode_N
        self.data = mixture
        self.labels = labs.long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, k):
        return self.data[k, ...], self.labels[k]

    def embed(self, dim=2, method="tsne", **kwargs):
        if dim not in [2, 3]:
            raise (ValueError("dim must be 2, or 3"))
        if dim == self.data.shape[-1]:
            return self.data
        if method == "tsne":
            embedder = TSNE(n_components=dim,
                            learning_rate="auto", init="pca", **kwargs)
        elif method == "pca":
            embedder = PCA(n_components=dim, **kwargs)
        else:
            raise (ValueError("Embedding %s not supported " % method))
        embedding = embedder.fit_transform(self.data)
        return embedding

    def visualize_embedding(self, dim=2, method="tsne", embedding=None, **kwargs):
        """Return a low-dimensional embedding of the data set.

        """
        sb.set(font_scale=2)
        if embedding is None:
            embedding = self.embed(dim=dim, method=method, **kwargs)
        cm = sb.color_palette("husl", len(np.unique(self.labels)))
        if dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       c=[cm[i] for i in self.labels])
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       embedding[:, 2], c=[cm[i] for i in self.labels])
        return fig

    def visualize_all(self, savedir, name):
        os.makedirs(savedir, exist_ok=True)
        try:
            fig_2d_tsne = self.visualize_embedding(dim=2, method="tsne")
            plt.title(name)
            fname = os.path.join(
                savedir, "%s_2d_tsne.png" % name)
            fig_2d_tsne.savefig(fname)
            plt.close(fig_2d_tsne)
            print("\tdone 2d tsne, saved in ", fname)
        except Exception as e:
            print("Exception for 2D-TSNE ", str(e))
            pass
        try:
            fig_3d_tsne = self.visualize_embedding(dim=3, method="tsne")
            plt.title(name)
            fname = os.path.join(
                savedir, "%s_3d_tsne.png" % name)
            fig_3d_tsne.savefig(fname)
            plt.close(fig_3d_tsne)
            print("\tdone 3d tsne, saved in ", fname)
        except Exception as e:
            print("Exception for 3D-TSNE ", str(e))
            pass
        try:
            fig_2d_pca = self.visualize_embedding(dim=2, method="pca")
            plt.title(name)
            fname = os.path.join(
                savedir, "%s_2d_pca.png" % name)
            fig_2d_pca.savefig(fname)
            plt.close(fig_2d_pca)
            print("\tdone 2d pca, saved in ", fname)
        except Exception as e:
            print("Exception for 2D-PCA ", str(e))
            pass
        try:
            fig_3d_pca = self.visualize_embedding(dim=3, method="pca")
            plt.title(name)
            fname = os.path.join(
                savedir, "%s_3d_pca.png" % name)
            fig_3d_pca.savefig(fname)
            plt.close(fig_3d_pca)
            print("\tdone 3d pca, saved in ", fname)
        except Exception as e:
            print("Exception for 3D-PCA ", str(e))
            pass


if __name__ == "__main__":
    import os
    from dynamics.utils import DEFAULT_DIST_KWARGS
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    for d in [64, 32, 16, 8, 4]:
        for k, v in DEFAULT_DIST_KWARGS.items():
            for mgm in [None, "rand", "randn", "beta"]:
                torch.manual_seed(314)
                np.random.seed(314)
                N = 100
                C = 8
                D = 3
                E = d
                M = k
                MA = v
                MK = {}
                LM = mgm
                LA = []
                CM = mgm
                CA = []
                GM = mgm
                GA = []
                if mgm == "beta":
                    GA = [0.5, 0.5]
                    LA = [0.5, 0.5]
                    CA = [0.5, 0.5]
                name = "M=%s;N=%s;A=%s;KW=%s;DI=%s;SD=%s;ED=%s;GM=%s;GA=%s;LM=%s;LA=%s;CM=%s;CA=%s" % (M,
                                                                                                       N,
                                                                                                       MA,
                                                                                                       MK,
                                                                                                       M,
                                                                                                       D,
                                                                                                       E,
                                                                                                       GM,
                                                                                                       GA,
                                                                                                       LM,
                                                                                                       LA,
                                                                                                       CM,
                                                                                                       CA
                                                                                                       )
                print("Trying ", name)
                kwargs = {
                    "modes": C,
                    "N": N,
                    "args": MA,
                    "kwargs": MK,
                    "distributions": M,
                    "source_dim": D,
                    "embedding_dim": E,
                    "loc_methods": LM,
                    "loc_args": LA,
                    "cov_methods": CM,
                    "cov_args": CA,
                    "gen_methods": GM,
                    "gen_args": GA
                }
                savedir = os.path.join("examples", "rm", M, "%s->%s" % (D, E))
                os.makedirs(savedir, exist_ok=True)
                try:
                    gm = RandomMixture(**kwargs)
                except Exception as e:
                    print("EXCEPTION ")
                    print(str(e))
                    #raise (e)
                    continue
