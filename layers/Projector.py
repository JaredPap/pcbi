# -*- coding: utf-8 -*-
# @Time    : 2023/8/12 下午9:56
# @Author  : Chen Mukun
# @File    : Projector.py
# @Software: PyCharm
# @desc    : 
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self,
                 in_feats):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_feats, in_feats//2),
            nn.ReLU(),
            nn.Linear(in_feats//2, in_feats)
        )

    def forward(self,
                features):
        return self.projector(features)
