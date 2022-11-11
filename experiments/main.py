import os
import copy
import tqdm

import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dynamics.argkeys as ak
from sklearn.metrics import roc_auc_score
from torchmetrics import AUROC
from dynamics.hooks.group_hook import GroupHook

DEFAULT = {
    "lr": 1e-06,
    "optimizer": "adam",
    "model": "linear_network",
    "model_args": """[
        128,
        [
            128,
            128,
            128
        ],
        128
    ]""",
    "model_kwargs": """{
        "output_activation": nn.Softmax
    }""",
    "dataset": "random_mixture",
    "dataset_args": "[]",
    "dataset_kwargs": """{
        "modes": 128,
        "N": 256,
        "args": [],
        "kwargs": {},
        "distributions": "MultivariateNormal",
        "source_dim": 2,
        "embedding_dim": 128,
        "loc_methods": "rand",
        "loc_args": [],
        "cov_methods": "rand",
        "cov_args": [],
        "gen_methods": None,
        "gen_args": None
    }""",
    "batch_size": 128,
    "seed": 314159,
    "epochs": 100,
    "k": 0,
    "num_folds": 10,
    "criterion": "ce",
    "output_root": "results",
    "name": "E009_test",
    "metrics": [
        "auc",
        "group_hooks"
    ],
    "autoencoder": False
}

USE = DEFAULT

if __name__ == "__main__":
    import json
    from dynamics.utils import make_serializable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ak.create_parser(USE)
    args = parser.parse_args()
    args = ak.resolve_args(args)
    ps_args = make_serializable(args.__dict__)
    print("ARGS ", ps_args)
    output_path = os.path.join(args.output_root, args.name)
    hookdir = os.path.join(output_path, "hooks")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(hookdir, exist_ok=True)
    json.dump(ps_args, open(
        os.path.join(output_path, "params.json"), "w"), indent=4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = args.model(*args.model_args, **args.model_kwargs).to(device)
    hook = GroupHook(model, layer_names=["Linear", "Conv2d", "Conv3d"])
    dataset = args.dataset(*args.dataset_args, **args.dataset_kwargs)
    try:
        datadir = os.path.join(output_path, "data")
        print("Data dir ", datadir)
        os.makedirs(datadir, exist_ok=True)
        dataset.visualize_all(datadir, args.name)
    except Exception as e:
        print("Exception when visualizing ", str(e))
        pass
    kf = StratifiedKFold(n_splits=args.num_folds,
                         shuffle=True, random_state=args.seed)
    train_idx = test_idx = None
    for kk, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)), dataset.labels)):
        if kk == args.k:
            break
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=True)
    optimizer = args.optimizer(model.parameters(), lr=args.lr)
    criterion = args.criterion()
    unique_labels = np.unique(dataset.labels)
    #print("Uniques ! ", unique_labels)
    rows = []

    for epoch in range(args.epochs):
        group_stats = dict()
        print("Beggining epoch {epoch}".format(epoch=epoch))
        row = copy.deepcopy(args.__dict__)
        model.train()
        train_losses = []
        test_losses = []
        train_metrics = {("train_%s" % metric): []
                         for metric in args.metrics if metric != "group_hooks"}
        test_metrics = {("test_%s" % metric): []
                        for metric in args.metrics if metric != "group_hooks"}
        group_stats = None
        epoch_stats = None
        for batch_i, (x, y) in tqdm(enumerate(train_dataloader)):
            x = x.to(device)
            y_o = y.clone()
            if args.autoencoder:
                y = x.clone()
            else:
                y = y.to(device)
            #print("train ", batch_i, " y_unique ", y.unique())
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            for metric in args.metrics:
                if metric == "auc" and not args.autoencoder:
                    #print("WERE HERE")
                    # score = roc_auc_score(y.detach().cpu().numpy(),
                    #                      yhat.detach().cpu().numpy(),
                    #                      multi_class="ovr",
                    #                      labels=unique_labels)
                    auroc = AUROC(num_classes=len(unique_labels))
                    score = auroc(yhat, y).item()
                    #print("BUT IT DIDN'T WORK")
                    train_metrics["train_%s" % metric].append(score)
                elif metric == "group_hooks":
                    hook.parse_groups(y_o)
                    if epoch_stats is None:
                        epoch_stats = copy.deepcopy(hook.group_stats)
                    else:
                        for group, stat_dict in hook.group_stats.items():
                            if group not in epoch_stats.keys():
                                epoch_stats[group] = dict()
                            for stat_type, module_dict in stat_dict.items():
                                if stat_type not in epoch_stats[group].keys():
                                    epoch_stats[group][stat_type] = dict()
                                for module, stat_list in module_dict.items():
                                    if module not in epoch_stats[group][stat_type].keys():
                                        epoch_stats[group][stat_type][module] = [
                                        ]
                                    epoch_stats[group][stat_type][module].extend(
                                        stat_list)
                    # hook.write(os.path.join(output_path, "hooks",
                    #           "e%d_b%d.pkl" % (epoch, batch_i)))
                    hook.clear()
        print("\tAverage Training Loss {loss}".format(
            loss=np.mean(train_losses)))
        # hook.clear()
        model.eval()
        for metric in args.metrics:
            if "group_hooks" in metric:
                pkl.dump(epoch_stats, open(os.path.join(hookdir,
                                                        "e%d.pkl" % (epoch)), "wb"))
            else:
                for k, v in train_metrics.items():
                    row[k] = np.mean(v)
                    print("\t Average {metric} {val}".format(
                        metric=k, val=np.mean(v)))
        for batch_i, (x, y) in tqdm(enumerate(test_dataloader)):
            x = x.to(device)
            y = y.to(device)
            if args.autoencoder:
                y = x.clone()
            #print("test ", batch_i, " y_unique ", y.unique)
            yhat = model(x)
            loss = criterion(yhat, y)
            test_losses.append(loss.item())
            for metric in args.metrics:
                if metric == "auc" and not args.autoencoder:
                    # score = roc_auc_score(
                    #    y.detach().cpu().numpy(), yhat.detach().cpu().numpy(), multi_class="ovr", labels=unique_labels)
                    auroc = AUROC(num_classes=len(unique_labels))
                    score = auroc(yhat, y).item()
                    test_metrics["test_%s" % metric].append(score)
                elif metric == "group_hooks":
                    pass
        print("\tAverage Test Loss {loss}".format(loss=np.mean(test_losses)))
        row["epoch"] = epoch
        row["train_loss_mean"] = np.mean(train_losses)
        row["test_loss_mean"] = np.mean(test_losses)
        for metric in args.metrics:
            os.makedirs(os.path.join(output_path, "hooks"), exist_ok=True)
            if "group_hooks" in metric:
                pass
                # pkl.dump(group_stats,
                #         open(os.path.join(output_path, "hooks", "epoch_%d.pkl" % epoch), "wb"))
            else:
                for k, v in test_metrics.items():
                    row[k] = np.mean(v)
                    print("\t Average {metric} {val}".format(
                        metric=k, val=np.mean(v)))
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_path, "results.csv"))
