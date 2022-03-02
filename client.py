# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 11:19
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import copy
import sys
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('../')
from torch import nn
from data_process import nn_seq_wind, device
from ScaffoldOptimizer import ScaffoldOptimizer


def train(ann, server):
    ann.train()
    Dtr, Dte = nn_seq_wind(ann.name, ann.B)
    ann.len = len(Dtr)
    print('training...')
    loss_function = nn.MSELoss().to(device)
    loss = 0
    x = copy.deepcopy(ann)
    optimizer = ScaffoldOptimizer(ann.parameters(), lr=ann.lr, weight_decay=1e-4)
    for epoch in range(ann.E):
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = ann(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(server.control, ann.control)

        print('epoch', epoch, ':', loss.item())
    # update c
    # c+ <- ci - c + 1/(E * lr) * (x-yi)
    # save ann
    temp = {}
    for k, v in ann.named_parameters():
        temp[k] = v.data.clone()

    for k, v in x.named_parameters():
        ann.control[k] = ann.control[k] - server.control[k] + (v.data - temp[k]) / (ann.E * ann.lr)
        ann.delta_y[k] = temp[k] - v.data
        ann.delta_control[k] = ann.control[k] - x.control[k]

    return ann


def test(ann):
    ann.eval()
    Dtr, Dte = nn_seq_wind(ann.name, ann.B)
    pred = []
    y = []
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))
