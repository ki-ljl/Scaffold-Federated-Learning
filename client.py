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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.append('../')
from torch import nn
from get_data import nn_seq_wind, device
from ScaffoldOptimizer import ScaffoldOptimizer


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(ann, server):
    ann.train()
    Dtr, Val, Dte = nn_seq_wind(ann.name, ann.B)
    ann.len = len(Dtr)
    print('training...')
    loss_function = nn.MSELoss().to(device)
    x = copy.deepcopy(ann)
    optimizer = ScaffoldOptimizer(ann.parameters(), lr=ann.lr, weight_decay=1e-4)
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(ann.E)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = ann(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(server.control, ann.control)
        lr_step.step()
        # validation
        val_loss = get_val_loss(ann, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(ann)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        ann.train()

    ann = copy.deepcopy(best_model)
    # update c
    # c+ <- ci - c + 1/(steps * lr) * (x-yi)
    # save ann
    temp = {}
    for k, v in ann.named_parameters():
        temp[k] = v.data.clone()

    for k, v in x.named_parameters():
        local_steps = ann.E * len(Dtr)
        ann.control[k] = ann.control[k] - server.control[k] + (v.data - temp[k]) / (local_steps * ann.lr)
        ann.delta_y[k] = temp[k] - v.data
        ann.delta_control[k] = ann.control[k] - x.control[k]

    return ann


def test(ann):
    ann.eval()
    Dtr, Val, Dte = nn_seq_wind(ann.name, ann.B)
    pred = []
    y = []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))
