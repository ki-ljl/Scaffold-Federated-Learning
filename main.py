# -*- coding:utf-8 -*-
"""
@Time：2022/05/05 12:57
@Author：KI
@File：main.py
@Motto：Hungry And Humble
"""
from data_process import clients_wind
from server import Scaffold


def main():
    K, C, E, B, r = 10, 0.5, 10, 50, 10
    input_dim = 28
    lr = 0.08
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r, 'clients': clients_wind,
               'input_dim': input_dim, 'lr': lr}
    scaffold = Scaffold(options)
    scaffold.server()
    scaffold.global_test()


if __name__ == '__main__':
    main()
