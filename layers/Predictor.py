# -*- coding: utf-8 -*-
# @Time    : 2023/8/12 下午9:59
# @Author  : Chen Mukun
# @File    : Predictor.py
# @Software: PyCharm
# @desc    : 

import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, in_feats,
                 out_feats,
                 dropout=0.4
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_feats, 64)
        self.activation = nn.GELU()
        self.batch_normal = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, out_feats)

    def forward(self,
                features):
        emb = self.dropout(features)
        emb = self.batch_normal(self.activation(self.linear1(emb)))
        emb = self.linear2(emb)
        return emb