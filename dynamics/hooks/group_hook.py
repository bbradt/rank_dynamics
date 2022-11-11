import os
import torch
import pickle as pk
from dynamics.hooks.model_hook import ModelHook
from torch.nn.grad import conv3d_weight
import numpy as np


class GroupHook(ModelHook):
    def __init__(self, model, verbose=False, layer_names=None, register_self=True, save=True):
        super(GroupHook, self).__init__(
            model, verbose, layer_names, register_self, save)
        self.clear()

    def clear(self):
        super(GroupHook, self).clear()
        self.group_stats = dict()

    def write(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pk.dump(self.group_stats, open(filename, "wb"))

    def parse_groups(self, labels):
        uniques = labels.unique()
        for module in self.keys:
            real_module = self.modules[module]
            forward = self.forward_stats[module]
            act = forward["input"][-1][0]
            actr = act
            if "Conv3d" in module:
                actr = act.permute(1, 0, 2, 3, 4)
                actr = actr.view(act.shape[1], act.shape[0], int(
                    np.prod(act.shape[2:])))
            backward = self.backward_stats[module]
            delta = backward["output"][-1]
            deltar = delta
            if "Conv3d" in module:
                deltar = delta.permute(1, 0, 2, 3, 4)
                deltar = deltar.view(
                    delta.shape[1], delta.shape[0], int(np.prod(delta.shape[2:])))
            W = self.modules[module].weight
            for unique in uniques:
                # 16 x (38 x 38 x 38)
                idx = labels.view(labels.shape[0], -1) == unique
                u_act = torch.zeros_like(actr, device=actr.device)
                u_delta = torch.zeros_like(deltar, device=deltar.device)
                # 16 x 28 x 38 x 38 x 38
                u_act[idx.flatten(), :] = actr[idx.flatten(), ...]
                u_delta[idx.flatten(), :] = deltar[idx.flatten(), ...]
                if "Conv3d" in module:
                    u_act = u_act.view(
                        act.shape[1], act.shape[0], act.shape[2], act.shape[3], act.shape[4])
                    u_act = u_act.permute(1, 0, 2, 3, 4)
                    u_delta = u_delta.view(
                        delta.shape[1], delta.shape[0], delta.shape[2], delta.shape[3], delta.shape[4])
                    u_delta = u_delta.permute(1, 0, 2, 3, 4)
                u_grad = None

                if "Conv3d" in module:
                    u_grad = conv3d_weight(u_act, W.shape, u_delta,
                                           stride=real_module.stride,
                                           padding=real_module.padding,
                                           dilation=real_module.dilation,
                                           groups=real_module.groups)
                else:
                    #u_grad = u_act @ u_delta.t()
                    u_grad = (u_act.t() @ u_delta).t()
                #del u_act
                #del u_delta
                #_,S,_ = torch.linalg.svd(u_grad.squeeze())
                #del u_grad

                #u_grad = ( u_delta @ u_act ) @ W
                self.add_stat(unique.item(), "act",
                              module, u_act.detach().cpu())
                self.add_stat(unique.item(), "delta",
                              module, u_delta.detach().cpu())
                self.add_stat(unique.item(), "grad",
                              module, u_grad.detach().cpu())
                #self.add_stat(unique.item(), "grad_S", S.detach().cpu())
                #self.add_stat(unique.item(), "delta", u_delta)
                #self.add_stat(unique.item(), "grad", W.grad)
        return self.group_stats

    def add_stat(self, group, key, layer, stat):
        if group not in self.group_stats.keys():
            self.group_stats[group] = dict()
        if key not in self.group_stats[group].keys():
            self.group_stats[group][key] = dict()
        if layer not in self.group_stats[group][key].keys():
            self.group_stats[group][key][layer] = []
        self.group_stats[group][key][layer].append(stat)
