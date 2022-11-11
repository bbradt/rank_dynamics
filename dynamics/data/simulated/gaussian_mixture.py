import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class GaussianMixture(Dataset):
    def __init__(self,
                 Ns=[64],
                 locs=[[0, 0, 0]],
                 covariances=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                 embedding_dim=2,
                 labels=None
                 ):
        super(GaussianMixture, self).__init__()
        Nss = np.cumsum(Ns)
        mixture = torch.zeros((max(Nss), embedding_dim))
        Nlast = 0
        labs = torch.zeros((max(Nss),))
        for ii, (N, loc, cov) in enumerate(zip(Nss, locs, covariances)):
            distribution = torch.distributions.MultivariateNormal(torch.Tensor(loc),
                                                                  torch.Tensor(cov))
            sample = distribution.sample((N-Nlast, ))
            Uv, Sv, Vv = torch.linalg.svd(sample, full_matrices=False)
            Sv = torch.diag_embed(Sv)
            if len(loc) < embedding_dim:
                # project into higher dimensional space
                offset_min = 0
                offset_max = embedding_dim - Vv.shape[0] - 1
                offset = max(np.random.randint(offset_min, offset_max), 0)
                #offset = 0
                Uu = torch.zeros((N-Nlast, embedding_dim))
                Su = torch.zeros((embedding_dim, embedding_dim))
                Vu = torch.zeros((embedding_dim, embedding_dim))
                Uu[:, offset:(Uv.shape[1]+offset)] = Uv
                Vu[offset:(Vv.shape[0]+offset),
                   offset:(Vv.shape[1]+offset)] = Vv
                Su[offset:(Sv.shape[0]+offset),
                   offset:(Sv.shape[1]+offset)] = Sv
                sample = Uu @ Su @ Vu
            elif len(loc) > embedding_dim:
                Sz = Sv[:embedding_dim, :embedding_dim]
                sample = Uv[:,
                            :embedding_dim] @ Sz @ Vv[:embedding_dim, :embedding_dim]
            mixture[Nlast:N, :] += sample
            labs[Nlast:N] = ii
            Nlast = N
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

    @staticmethod
    def make_covariance(method, dim, *args, **kwargs):
        if method in ["rand", "randn"]:
            A = getattr(np.random, method)(dim, dim)
        elif method in ["ranf"]:
            A = getattr(np.random, method)((dim, dim))
        else:
            A = getattr(np.random, method)(*args, size=(dim, dim), **kwargs)
        C = A.T @ A
        C = C / np.max(C)
        np.fill_diagonal(C, 1)
        return C

    @staticmethod
    def make_means(method, dim, *args, **kwargs):
        if method in ["rand", "randn"]:
            A = getattr(np.random, method)(dim)
        elif method in ["ranf"]:
            A = getattr(np.random, method)((dim,))
        else:
            A = getattr(np.random, method)(*args, size=(dim,), **kwargs)
        return A.flatten()

    @staticmethod
    def make_kwargs(N, num_classes, source_dim, embedding_dim, method="randn", method_args=[], method_kwargs={}):
        kwargs = dict()
        if type(N) is int:
            kwargs["Ns"] = [N for _ in range(num_classes)]
        elif type(N) is list:
            kwargs["Ns"] = N[:num_classes]
        if type(method) is str and type(source_dim) is int:
            kwargs["locs"] = [GaussianMixture.make_means(
                method, source_dim, *method_args, **method_kwargs) for _ in range(num_classes)]
        elif type(method) is str and type(source_dim) is list:
            kwargs["locs"] = [GaussianMixture.make_means(
                method, s, *method_args, **method_kwargs) for s in source_dim]
        elif type(method) is list and type(source_dim) is int:
            kwargs["locs"] = [GaussianMixture.make_means(
                m, source_dim, *method_args, **method_kwargs) for m in method]
        elif type(method) is list and type(source_dim) is list:
            kwargs["locs"] = [GaussianMixture.make_means(
                m, s, *method_args, **method_kwargs) for m, s in zip(method, source_dim)]
        else:
            raise (ValueError(
                "Type of method and source dim may only be list or string"))
        if type(method) is str and type(source_dim) is int:
            kwargs["covariances"] = [GaussianMixture.make_covariance(
                method, source_dim, *method_args, **method_kwargs) for _ in range(num_classes)]
        elif type(method) is str and type(source_dim) is list:
            kwargs["covariances"] = [GaussianMixture.make_covariance(
                method, s, *method_args, **method_kwargs) for s in source_dim]
        elif type(method) is list and type(source_dim) is int:
            kwargs["covariances"] = [GaussianMixture.make_covariance(
                m, source_dim, *method_args, **method_kwargs) for m in method]
        elif type(method) is list and type(source_dim) is list:
            kwargs["covariances"] = [GaussianMixture.make_covariance(
                m, s, *method_args, **method_kwargs) for m, s in zip(method, source_dim)]
        else:
            raise (ValueError(
                "Type of method and source dim may only be list or string"))
        kwargs["embedding_dim"] = embedding_dim
        return kwargs


if __name__ == "__main__":
    import os
    torch.manual_seed(314)
    np.random.seed(314)
    N = 1024
    C = 8
    D = 3
    E = 3
    M = "beta"
    MA = []
    MK = {}
    name = "%s_%s_%s_%sc_%s->%s" % (M, str(MA), str(MK), C, D, E)
    dataset_kwargs = GaussianMixture.make_kwargs(
        N, C, D, E, M, method_args=MA, method_kwargs=MK)
    savedir = os.path.join("examples", "gm", M)
    os.makedirs(savedir, exist_ok=True)
    gm = GaussianMixture(**dataset_kwargs)
    fig_2d_tsne = gm.visualize_embedding(dim=2, method="tsne")
    plt.title(name)
    fig_2d_tsne.savefig(os.path.join(savedir, "gm_%s_2d_tsne.png" % name))
    plt.close(fig_2d_tsne)
    print("Done 2d tsne")
    fig_3d_tsne = gm.visualize_embedding(dim=3, method="tsne")
    plt.title(name)
    fig_3d_tsne.savefig(os.path.join(savedir, "gm_%s_3d_tsne.png" % name))
    plt.close(fig_3d_tsne)
    print("Done 3d tsne")
    fig_2d_pca = gm.visualize_embedding(dim=2, method="pca")
    plt.title(name)
    fig_2d_pca.savefig(os.path.join(savedir, "gm_%s_2d_pca.png" % name))
    plt.close(fig_2d_pca)
    print("Done 2d pca")
    fig_3d_pca = gm.visualize_embedding(dim=3, method="pca")
    plt.title(name)
    fig_3d_pca.savefig(os.path.join(savedir, "gm_%s_3d_pca.png" % name))
    plt.close(fig_3d_pca)
    print("Done 3d pca")
