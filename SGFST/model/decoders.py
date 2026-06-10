from typing import Optional, Literal, List
from model.layers import GraphConvolution

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDecoder(nn.Module):
    """
    Decoder that computes the cosine similarity between node embeddings.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input node embeddings.
    """

    def __init__(self, dropout: float = 0.1, zero_diag: bool = True) -> None:
        super(CosineDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.zero_diag = zero_diag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CosineDecoder.

        Parameters
        ----------
        x:
            Input node embeddings (dim: n_nodes x input_dim).

        Returns
        ----------
        adj_reconstructed:
            Reconstructed adjacency matrix based on cosine similarity
            (dim: n_nodes x n_nodes).
        """
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        S = x @ x.t()
        if self.zero_diag:
            S = S - torch.diag_embed(torch.diag(S))
        return S


class OmicsDecoder(nn.Module):
    """
    Decoder that reconstructs omics data from node embeddings.

    Parameters
    ----------
    nfeat:
        Dimensionality of the input omics data.
    nhid1:
        Dimensionality of the first hidden layer.
    nhid2:
        Dimensionality of the second hidden layer (input node embeddings).
    """

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, use_bn: bool = False) -> None:
        super(OmicsDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU()
        )

        self.gc1 = GraphConvolution(nhid2, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid1)
        self.norm1 = nn.BatchNorm1d(nhid1) if use_bn else nn.Identity()
        self.norm2 = nn.BatchNorm1d(nhid1) if use_bn else nn.Identity()
        self.act = nn.ReLU()

        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        # self.mean = nn.Linear(nhid1, nfeat)
        self.mean = GraphConvolution(nhid1, nfeat)

        self.DispAct = lambda x: torch.clamp(nn.functional.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb: torch.Tensor, adj: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the OmicsDecoder.

        Parameters
        ----------
        emb:
            Input node embeddings (dim: n_nodes x nhid2).
        adj:
            Adjacency matrix of the graph (dim: n_nodes x n_nodes).

        Returns
        ----------
        pi:
            Dropout probabilities for each feature (dim: n_nodes x nfeat).
        disp:
            Dispersion parameters for each feature (dim: n_nodes x nfeat).
        mean:
            Mean parameters for each feature (dim: n_nodes x nfeat).
        """
        # x = self.gc1(emb, adj)
        # x = self.norm1(x)
        # x = self.act(x)
        #
        # x = self.gc2(x, adj)
        # x = self.norm2(x)
        # x = self.act(x)
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x, adj))
        return [pi, disp, mean]
