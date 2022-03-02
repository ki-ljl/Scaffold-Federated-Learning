# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 11:22
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""
import copy
import random
import sys

import numpy as np
import torch

sys.path.append('../')
from model import ANN
from data_process import device, clients_wind
from client import train, test


class Scaffold:
    def __init__(self, options):
        self.C = options['C']
        self.E = options['E']
        self.B = options['B']
        self.K = options['K']
        self.r = options['r']
        self.input_dim = options['input_dim']
        self.lr = options['lr']
        self.clients = options['clients']
        self.nn = ANN(input_dim=self.input_dim, name='server', B=self.B, E=self.E, lr=self.lr).to(
            device)
        # self.control = torch.zeros_like(self.nn.named_parameters)
        for k, v in self.nn.named_parameters():
            self.nn.control[k] = torch.zeros_like(v.data)
            self.nn.delta_control[k] = torch.zeros_like(v.data)
            self.nn.delta_y[k] = torch.zeros_like(v.data)
        self.nns = []
        for i in range(self.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            temp.control = copy.deepcopy(self.nn.control)  # ci
            temp.delta_control = copy.deepcopy(self.nn.delta_control)  # ci
            temp.delta_y = copy.deepcopy(self.nn.delta_y)
            self.nns.append(temp)

    def server(self):
        for t in range(self.r):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.C * self.K), 1])
            index = random.sample(range(0, self.K), m)
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0.0
        for j in index:
            # normal
            s += self.nns[j].len
        # compute
        x = {}
        c = {}
        # init
        for k, v in self.nns[0].named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                x[k] += self.nns[j].delta_y[k] / len(index)  # averaging
                c[k] += self.nns[j].delta_control[k] / len(index)  # averaging

        # update x and c
        for k, v in self.nn.named_parameters():
            v.data += x[k].data  # lr=1
            self.nn.control[k].data += c[k].data * (len(index) / self.K)

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                new_params.data = old_params.data.clone()

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.nns[k], self.nn)

    def global_test(self):
        model = self.nn
        model.eval()
        c = clients_wind
        for client in c:
            model.name = client
            test(model)


if __name__ == '__main__':
    K, C, E, B, r = 10, 0.5, 10, 50, 10
    input_dim = 28
    lr = 0.01
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r, 'clients': clients_wind,
               'input_dim': input_dim, 'lr': lr}
    scaffold = Scaffold(options)
    scaffold.server()
    scaffold.global_test()
