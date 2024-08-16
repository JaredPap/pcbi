# -*- coding: utf-8 -*-
# @Time    : 2023/11/29 8:30 下午
# @Author  : Chen Mukun
# @File    : HMGEncoder.py
# @Software: PyCharm
# @desc    :


import math
import dgl

import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2):
        """ Initializes the Heterogeneous Graph Transformer (HGT) Layer.

        Args:
            in_dim (int): Input dimension size.
            out_dim (int): Output dimension size.
            node_dict (dict): A dictionary mapping node types to indices.
            edge_dict (dict): A dictionary mapping edge types to indices.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate for regularization.
        """
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads  # Dimension per head
        self.sqrt_dk = math.sqrt(self.d_k)

        # Dropout layers for node and edge features
        self.dropout_layer_n = nn.Dropout(p=dropout)
        self.dropout_layer_e = nn.Dropout(p=dropout)

        # Linear transformations for Q, K, V for nodes
        self.n_k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(node_dict))])
        self.n_q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(node_dict))])
        self.n_v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(node_dict))])

        # Linear transformations for Q, K, V for edges
        self.e_k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(edge_dict))])
        self.e_q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(edge_dict))])
        self.e_v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(len(edge_dict))])

        # Final linear transformations for node and edge updates
        self.W_n_delta = nn.ModuleList([nn.Linear(2 * out_dim, out_dim) for _ in node_dict])
        self.W_e_delta = nn.ModuleList([nn.Linear(2 * out_dim, out_dim) for _ in edge_dict])

        self.n_linear = nn.Linear(self.out_dim, self.out_dim)
        self.e_linear = nn.Linear(self.out_dim, self.out_dim)

        self.W_n_delta = nn.ModuleList()
        self.W_e_delta = nn.ModuleList()

        self.drop = nn.Dropout(dropout)

        self.type_mapping = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(len(node_dict) + len(edge_dict))])

        self.activate = torch.nn.LeakyReLU()

    def forward(self,
                G: dgl.DGLGraph,
                node_ids_dict,
                edge_ids_dict):
        """ Forward pass of the HGT layer.

                Args:
                    G (dgl.DGLGraph): The graph.
                    node_ids_dict (dict): Dictionary of node type to node ID mapping.
                    edge_ids_dict (dict): Dictionary of edge type to edge ID mapping.
        """

        # QVK
        for ntype in G.ntypes:
            # num_node * num_head * d_k
            if G.num_nodes(ntype) == 0:
                continue
            # Compute Q, K, V for nodes
            G.nodes[ntype].data["Q"] = self.type_mapping[self.node_dict[ntype]](
                self.n_q_linears[self.node_dict[ntype]](G.nodes[ntype].data['h'])).view(-1, self.n_heads, self.d_k)
            G.nodes[ntype].data["K"] = self.type_mapping[self.node_dict[ntype]](
                self.n_k_linears[self.node_dict[ntype]](G.nodes[ntype].data['h'])).view(-1, self.n_heads, self.d_k)
            G.nodes[ntype].data["V"] = self.type_mapping[self.node_dict[ntype]](
                self.n_v_linears[self.node_dict[ntype]](G.nodes[ntype].data['h'])).view(-1, self.n_heads, self.d_k)

        for srctype, etype, dsttype in G.canonical_etypes:
            # num_edge * num_head * d_k
            if G.num_edges(etype) == 0:
                continue
            G.edges[etype].data["Q"] = self.type_mapping[self.num_types + self.edge_dict[etype]](
                self.e_q_linears[self.edge_dict[etype]](
                    G.edges[etype].data['h'] +
                    G.nodes[srctype].data['h'][G.edges(etype=etype)[0]]
                )
            ).view(-1, self.n_heads, self.d_k)
            G.edges[etype].data["K"] = self.type_mapping[self.num_types + self.edge_dict[etype]](
                self.e_k_linears[self.edge_dict[etype]](
                    G.edges[etype].data['h'] +
                    G.nodes[srctype].data['h'][G.edges(etype=etype)[0]]
                )
            ).view(-1, self.n_heads, self.d_k)
            G.edges[etype].data["V"] = self.type_mapping[self.num_types + self.edge_dict[etype]](
                self.e_v_linears[self.edge_dict[etype]](
                    G.edges[etype].data['h'] +
                    G.nodes[srctype].data['h'][G.edges(etype=etype)[0]]
                )
            ).view(-1, self.n_heads, self.d_k)

        # Perform homogeneous graph transformations and message passing
        ho: dgl.DGLGraph = dgl.to_homogeneous(G, ndata=['Q', 'K', 'V'], edata=['Q', 'K', 'V'])
        lg: dgl.DGLGraph = ho.line_graph(shared=True)

        # node message passing:
        # Apply edge softmax for attention mechanism
        # "att" head * 1
        ho.apply_edges(fn.v_dot_e("Q", "K", "att"))
        ho.edata["att"] = edge_softmax(ho, ho.edata["att"] / self.sqrt_dk, norm_by="dst")
        # "V"   head * d_k
        # "t"   head * d_k
        # "m"   head * d_k
        # Apply updates to node features
        ho.update_all(
            message_func=lambda edges: {"t": edges.data["att"] * edges.data["V"]},
            reduce_func=fn.sum("t", "m"),
            apply_node_func=lambda nodes: {'m': nodes.data["m"].view(-1, self.out_dim)}
        )
        ho.apply_nodes(lambda nodes: {'m': nodes.data["m"].view(-1, self.out_dim)})

        # edge message passing:
        # v_i
        lg.add_nodes(ho.num_nodes(), data=ho.ndata)
        lg.add_edges(ho.edges()[0] + (lg.num_nodes() - ho.num_nodes()), lg.nodes()[:lg.num_nodes() - ho.num_nodes()])
        # Apply edge softmax for attention mechanism
        # "att" head * 1
        lg.apply_edges(fn.v_dot_u("Q", "K", "att"))
        # print("q*k", lg.edata["att"].shape)
        lg.edata["att"] = edge_softmax(lg, lg.edata["att"] / self.sqrt_dk, norm_by="dst")
        # "att" head * 1
        # "V"   head * d_k
        # "t"   head * d_k
        # "m"   head * d_k
        # Apply updates to edge features
        lg.update_all(message_func=fn.u_mul_e("V", "att", "t"),
                      reduce_func=fn.sum("t", "m"),
                      apply_node_func=lambda nodes: {'m': nodes.data["m"].view(-1, self.out_dim)})

        # update function
        for ntype in G.ntypes:
            G.nodes[ntype].data["h"] = self.dropout_layer_n(self.activate(
                self.W_n_delta[self.node_dict[ntype]](
                    torch.cat(tensors=[G.nodes[ntype].data["h"],
                                       self.n_linear(ho.ndata['m'][node_ids_dict[ntype][0]:node_ids_dict[ntype][1]])
                                       ], dim=1)
                )
            ))
        for etype in G.etypes:
            if G.num_edges(etype) == 0:
                continue
            G.edges[etype].data["h"] = self.dropout_layer_e(self.activate(
                self.W_e_delta[self.edge_dict[etype]](
                    torch.cat(tensors=[G.edges[etype].data["h"],
                                       self.n_linear(lg.ndata['m'][edge_ids_dict[etype][0]:edge_ids_dict[etype][1]])
                                       ], dim=1)
                )
            ))
        return G
