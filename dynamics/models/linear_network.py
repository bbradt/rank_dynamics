import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNetwork(nn.Module):
    def __init__(self,
                 input_features,
                 hidden_layers,
                 output_features,
                 activation_function=nn.Identity,
                 output_activation=nn.Identity,
                 bias=True
                 ):
        super(LinearNetwork, self).__init__()
        last_feature = input_features
        hidden_layers.append(output_features)
        self.layers = []
        self.activations = []
        for i, next_feature in enumerate(hidden_layers):
            layer = nn.Linear(last_feature, next_feature, bias=bias)
            setattr(self, "fc%d" % i, layer)
            self.layers.append("fc%d" % i)
            if i < len(hidden_layers) - 1:
                act = activation_function()
            elif output_activation is not None:
                act = output_activation(dim=1)
            else:
                act = nn.Identity()
            setattr(self, "act%d" % i, act)
            self.activations.append("act%d" % i)
            last_feature = next_feature

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            layer = getattr(self, layer)
            act = getattr(self, act)
            x = layer(x)
            x = act(x)
        return x


if __name__ == "__main__":
    import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100000
    M = 1024
    H = []
    B = 1000
    E = 1999
    X = torch.randn(N, M).to(device)
    V = torch.randn(N, M).to(device)
    criterion = nn.MSELoss()
    model = LinearNetwork(M, H, M)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    for i in range(E):
        pbar = tqdm.tqdm(range(0, N, B))
        losses = []
        for ii in pbar:
            opt.zero_grad()
            Xx = X[ii:(ii+B), ...]
            Xhat = model(Xx)
            loss = criterion(Xhat, Xx)
            pbar.set_postfix(dict(epoch=i, loss=loss.item()))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            Vhat = model(V)
            vloss = criterion(Vhat, V)
            print("Loss ", torch.mean(torch.Tensor(losses)).item(), vloss.item())
