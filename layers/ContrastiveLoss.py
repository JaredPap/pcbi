# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 下午6:47
# @Author  : Chen Mukun
# @File    : ContrastiveLoss.py
# @Software: PyCharm
# @desc    :

import torch.nn as nn
import torch
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self,
                 device,
                 temperature: float) -> None:
        super().__init__()
        """ Initialize the Contrastive Loss module.

                Args:
                    device (torch.device): The device tensors will be transferred to (CPU or GPU).
                    temperature (float): A temperature scaling factor to apply to the logits.
        """
        self.device = device
        self.loss_computer = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self,
                em1,
                em2):
        """ Forward pass of the contrastive loss calculation.
                Args:
                    em1 (Tensor): Embeddings from the first set.
                    em2 (Tensor): Embeddings from the second set.
                    em1.shape[0] <= em2.shape[0]
                Returns:
                    Tensor: The computed loss as a scalar.
        """
        # Calculate the sizes of two sets of embeddings
        m = em1.shape[0]
        n = em2.shape[0]
        if m > n:
            em1, em2 = em2, em1
            m, n = n, m

        # Normalize embeddings and compute the similarity matrix
        emb = torch.nn.functional.normalize(torch.cat([em1, em2]))
        similarity_matrix = torch.matmul(emb, emb.t()).to(self.device)

        # Create a mask for positive samples
        positives_mask = np.eye(m + n, k=n) + np.eye(m + n, k=-n)
        positives_mask = torch.from_numpy(positives_mask).to(self.device)
        positives = similarity_matrix[positives_mask.type(torch.bool)].view(m * 2, -1)

        # Create a mask for negative samples
        negatives_mask = 1 - positives_mask - torch.from_numpy(np.eye(m + n)).to(self.device)
        negatives_mask[m:n, :] = 0

        negatives_mask = torch.from_numpy(negatives_mask).type(torch.bool).to(self.device)
        negatives = similarity_matrix[negatives_mask].view(m * 2, -1)

        # Combine positives and negatives and scale by temperature
        logits = torch.cat((positives, negatives), dim=1) / self.temperature

        # Create labels for the positive samples; all zeros because positives are always first
        labels = torch.zeros(m * 2).long().to(self.device)

        # Calculate the contrastive loss
        loss = self.loss_computer(logits, labels)

        return loss

    def forward_(self,
                 em1,
                 em2):
        """ Forward pass of the contrastive loss calculation.
                Args:
                    em1 (Tensor): Embeddings from the first set.
                    em2 (Tensor): Embeddings from the second set.

                Returns:
                    Tensor: The computed loss as a scalar.
        """
        # Calculate the sizes of two sets of embeddings
        m = em1.shape[0]
        n = em2.shape[0]

        # Normalize embeddings and compute the similarity matrix
        emb = torch.nn.functional.normalize(torch.cat([em1, em2]))
        similarity_matrix = torch.matmul(emb, emb.t())

        # Create a mask for positive samples
        positives_mask = np.eye(m + n, k=n)
        if m > n:
            positives_mask[:m - n, :] = 0
        positives_mask = torch.from_numpy(positives_mask).type(torch.bool).to(self.device)
        positives = similarity_matrix[positives_mask].view(min(m, n), -1)

        # Create a mask for negative samples
        negatives_mask = 1 - (np.eye(m + n, k=n) + np.eye(m + n))
        if m <= n:
            negatives_mask[m:, :] = 0
        else:
            negatives_mask[:m - n, :] = 0
            negatives_mask[m:, :] = 0
        negatives_mask = torch.from_numpy(negatives_mask).type(torch.bool).to(self.device)
        negatives = similarity_matrix[negatives_mask].view(min(m, n), -1)

        # Combine positives and negatives and scale by temperature
        logits = torch.cat((positives, negatives), dim=1) / self.temperature

        # Create labels for the positive samples; all zeros because positives are always first
        labels = torch.zeros(min(m, n)).long().to(self.device)

        # Calculate the contrastive loss
        loss = self.loss_computer(logits, labels)

        return loss
