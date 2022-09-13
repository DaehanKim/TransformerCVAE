import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from data.util import *
import copy


def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def init_para_frompretrained(m, pm, share_para=False):
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)


def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """

    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f


def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    '''inverse multiquadratic kernel from https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py#L120
    fixed wrong expectation computation : res1 은 n(n-1)로 나눠져야 하고 res2는 n**2으로 나눠져야 하는데 각각 n-1, n으로 나눠졌음.

    '''
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1) / batch_size
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size / batch_size
        stats += res1 - res2

    return stats