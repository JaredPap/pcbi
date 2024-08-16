# -*- coding: utf-8 -*-
# @Time    : 2023/8/9 下午12:21
# @Author  : Chen Mukun
# @File    : SAGPooling.py
# @Software: PyCharm
# @desc    : 

import torch

from torch_geometric.nn import  GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK


class SAGPool(torch.nn.Module):

    def __init__(self, embed_dim, keep_ratio=0.8):
        """ Initializes the Self-Attention Graph Pooling (SAGPooling) module.
                Args:
                    embed_dim (int): The dimension of the input features.
                    keep_ratio (float): The ratio of the nodes to keep after pooling.
        """
        super().__init__()
        self.in_channels = embed_dim
        self.ratio = keep_ratio

        self.gnn = GraphConv(embed_dim, 1)
        # self.gnn = GCNConv(embed_dim, 1)
        self.select = SelectTopK(1, keep_ratio)
        self.connect = FilterEdges()

        self.gnn.reset_parameters()
        self.select.reset_parameters()

    def forward(self, input_feature, adj, graph_indicator, edge_attr=None):
        """ Forward pass for the SAGPool layer.
                Args:
                    input_feature (Tensor): Node features.
                    adj (Tensor): Adjacency matrix or edge list.
                    graph_indicator (Tensor): Batch vector which assigns each node to a graph.
                    edge_attr (Tensor, optional): Edge attributes.
                Returns:
                    tuple: Tuple containing new node features, edge indices, batch vector, and edge attributes.
        """
        # Calculate attention scores using a GNN layer
        if graph_indicator is None:
            graph_indicator = adj.new_zeros(input_feature.size(0))
        attn = self.gnn(input_feature, adj).squeeze()

        # Select top-k nodes based on the attention scores
        select_out = self.select(attn, graph_indicator)
        perm = select_out.node_index
        score = select_out.weight
        assert score is not None
        # Apply scores to the input features
        new_input_feature = input_feature[perm] * score.view(-1, 1)

        # Filter edges and update graph structure
        connect_out = self.connect(select_out, adj, edge_attr, graph_indicator)

        return new_input_feature, connect_out.edge_index, connect_out.batch, connect_out.edge_attr


class SAGPoolReadout(torch.nn.Module):
    def __init__(self,
                 embed_dim,
                 out_dim,
                 layer_num,
                 keep_ratio=0.8):
        """ Initializes a readout module that applies multiple SAGPool layers.

                Args:
                    embed_dim (int): Embedding dimension of input features.
                    out_dim (int): Output dimension of the final linear layer.
                    layer_num (int): Number of SAGPool layers to apply.
                    keep_ratio (float): Ratio of nodes to keep in each SAGPool layer.
        """
        super(SAGPoolReadout, self).__init__()
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.pools = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.pools.append(SAGPool(embed_dim, keep_ratio))
        self.linear = torch.nn.Linear(2 * embed_dim, out_dim)

    def forward(self,
                input_feature,
                adj,
                edge_attr=None,
                batch=None):
        """ Performs forward pass, applying SAGPool layers sequentially and then aggregating the results.

                Args:
                    input_feature (Tensor): Node features.
                    adj (Tensor): Adjacency matrix or edge list.
                    edge_attr (Tensor, optional): Edge attributes.
                    batch (Tensor, optional): Batch vector for node features.

                Returns:
                    Tensor: Output tensor after pooling and readout.
        """

        if batch is None:
            graph_indicator = adj.new_zeros(input_feature.size(0))
        else:
            graph_indicator = batch
        # Apply each SAGPool layer in sequence
        for l in range(self.layer_num):
            input_feature, adj, graph_indicator, edge_attr = self.pools[l](
                input_feature, adj, graph_indicator, edge_attr
            )
        # Perform global mean and max pooling
        readout = torch.cat(
            [
                global_mean_pool(input_feature, graph_indicator),
                global_max_pool(input_feature, graph_indicator),
            ],
            dim=1,
        )
        return self.linear(readout)


if __name__ == '__main__':
    # Example usage of the SAGPoolReadout model
    f = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 0, 1, 1, 1]]
    edge_index = [[0, 1], [2, 0], [1, 2], [0, 1]]
    edge_index_change = [[0, 2, 1, 0], [1, 0, 2, 1]]

    edge_index_t = torch.tensor(edge_index)
    edge_index_change_t = torch.tensor(edge_index_change, dtype=torch.long)
    f_t = torch.tensor(f, dtype=torch.float32)
    # 初始化输入 in_channel
    print(edge_index_t)
    print(edge_index_change_t)
    model = SAGPoolReadout(5, 2, 2)

    result = model(input_feature=f_t, adj=edge_index_change_t)
    print(result)
