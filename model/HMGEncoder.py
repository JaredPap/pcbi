# -*- coding: utf-8 -*-
# @Time    : 2023/11/29 8:30 下午
# @Author  : Chen Mukun
# @File    : HMGEncoder.py
# @Software: PyCharm
# @desc    :
import torch
import dgl
import torch.nn as nn
import numpy as np

from data.augment import Mol2HeteroGraph, input_info
from layers.SAGPooling import SAGPoolReadout
from layers.HGTLayer import HGTLayer


class HGT(nn.Module):
    def __init__(self,
                 in_dims,
                 hidden_dim,
                 node_dict,
                 edge_dict,
                 n_layers,
                 n_heads,
                 dropout=0.2):
        """ Initialize the Heterogeneous Graph Transformer (HGT) module.
            Args:
                    in_dims (dict): Input feature dimensions for each node type.
                    hidden_dim (int): Dimensionality of the hidden representations.
                    node_dict (dict): Dictionary mapping node types to indices.
                    edge_dict (dict): Dictionary mapping edge types to indices.
                    n_layers (int): Number of graph transformer layers.
                    n_heads (int): Number of attention heads.
                    dropout (float): Dropout rate.
        """
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleDict()
        for type, in_dim in in_dims.items():
            self.adapt_ws[type] = nn.Linear(in_dim, hidden_dim)
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, node_dict, edge_dict, n_heads,dropout=dropout))

    def forward(self, G):
        """ Forward pass for the HGT module.
                Args:
                    G (DGLGraph): The graph input.
                Returns:
                    DGLGraph: Updated graph with transformed node and edge features.
        """
        # Initialize node features from input data
        for ntype in G.ntypes:
            if G.num_nodes(ntype) == 0:
                continue
            G.nodes[ntype].data["h"] = self.adapt_ws[ntype](G.nodes[ntype].data["x"])
        # Initialize edge features
        for etype in G.etypes:
            if G.num_edges(etype) == 0:
                continue
            G.edges[etype].data["h"] = self.adapt_ws[etype](G.edges[etype].data["x"])

        # Convert the graph to a homogeneous format and get mapping counts
        ho, ntype_count, etype_count = dgl.to_homogeneous(G, return_count=True)
        node_id_mapping = np.cumsum(ntype_count)
        edge_id_mapping = np.cumsum(etype_count)

        node_ids_dict = {}
        edge_ids_dict = {}

        for i, ntype in enumerate(G.ntypes):
            if i == 0:
                node_ids_dict[ntype] = (0, node_id_mapping[i])
            else:
                node_ids_dict[ntype] = (node_id_mapping[i - 1], node_id_mapping[i])

        for i, etype in enumerate(G.etypes):
            if i == 0:
                edge_ids_dict[etype] = [0, edge_id_mapping[i]]
            else:
                edge_ids_dict[etype] = [edge_id_mapping[i - 1], edge_id_mapping[i]]

        # stack layers
        for i in range(self.n_layers):
            G = self.gcs[i](G, node_ids_dict, edge_ids_dict)
        return G


class HMGEncoder(nn.Module):
    def __init__(self,
                 in_dims,
                 hidden_feats,
                 out_feats,
                 n_heads,
                 n_layers,
                 node_dict,
                 edge_dict,
                 dropout=0.2):
        super(HMGEncoder, self).__init__()
        """ Initializes the Heterogeneous Molecular Graph (HMG) Encoder.
                Args:
                    in_dims (dict): Dictionary specifying the input feature dimensions for each node type.
                    hidden_feats (int): The number of hidden units at each layer.
                    out_feats (int): The dimensionality of the output features.
                    n_heads (int): The number of attention heads in multi-head attention mechanisms.
                    n_layers (int): The number of layers in the network.
                    node_dict (dict): Dictionary mapping node types to their respective indices.
                    edge_dict (dict): Dictionary mapping edge types to their respective indices.
                    dropout (float): Dropout rate used in the encoder for regularization.
        """
        self.node_dict = node_dict
        self.hgt = HGT(in_dims=in_dims,
                       hidden_dim=hidden_feats,
                       node_dict=node_dict,
                       edge_dict=edge_dict,
                       n_layers=n_layers,
                       n_heads=n_heads,
                       dropout=dropout)
        self.out = SAGPoolReadout(embed_dim=hidden_feats,
                                  out_dim=out_feats,
                                  layer_num=2)

    def forward(self,
                G,
                is_batch=False):
        """ Forward pass for processing graph data through the HMG Encoder.
                Args:
                    G (DGLGraph): The input graph or a batch of graphs.
                    is_batch (bool): Flag to indicate whether 'G' is a single graph or a batch of graphs.
                Returns:
                    Tensor: Output features from the final readout layer after encoding.
        """
        if is_batch:
            bg = G
            node_count = []
            # Collect the number of nodes for each type to handle batch data correctly
            for ntype in self.node_dict:
                node_count.append(bg.batch_num_nodes(ntype).unsqueeze(0))
            lengths_tensor = torch.cat(node_count, dim=0).sum(0).to(bg.device)
            indices = torch.arange(lengths_tensor.size(0)).to(bg.device)
            # batch mask for pooling layer
            batch = indices.repeat_interleave(lengths_tensor).to(bg.device)
            bg = self.hgt(bg)
            bg = dgl.to_homogeneous(bg, ndata=['h'], edata=['h'])
            adj = torch.stack(bg.adj_tensors('coo'), dim=0)

            # Apply readout to get the graph representation
            graph_read_out = self.out(input_feature=bg.ndata['h'], adj=adj, edge_attr=bg.edata['h'], batch=batch)
        else:
            g = self.hgt(G)
            g = dgl.to_homogeneous(g, ndata=['h'], edata=['h'])
            adj = torch.stack(g.adj_tensors('coo'), dim=0)
            graph_read_out = self.out(input_feature=g.ndata['h'], adj=adj, edge_attr=g.edata['h'])
        return graph_read_out

# Example of usage
if __name__ == '__main__':
    smi = "CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1"
    hidden_feats = 128  # Dimensionality of the hidden layer
    out_feats = 16  # Output feature dimensionality
    n_heads = 4  # Number of attention heads
    n_layers = 2  # Number of layers
    he = Mol2HeteroGraph(smi, False)
    node_dict, edge_dict, in_dims = input_info(he)
    model_mol = HMGEncoder(in_dims, hidden_feats, out_feats, n_heads, n_layers, node_dict, edge_dict)
    out = model_mol(he)
    print(out)
    bg = dgl.batch([he, he])
    out = model_mol(bg, True)
    print(out)

