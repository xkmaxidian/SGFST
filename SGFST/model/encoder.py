import torch
import torch.nn as nn

from model.layers import GraphConvolution


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder.

    Parameters
    ----------
    nfeat:
        Dimensionality of the input node features.
    nhid:
        Dimensionality of the hidden layer.
    nout:
        Dimensionality of the output node embeddings.
    dropout:
        Dropout rate for regularization.
    """

    def __init__(self, nfeat: int, nhid: int, nout: int, dropout: float) -> None:
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCNEncoder.

        Parameters
        ----------
        x:
            Input node features (dim: n_nodes x nfeat).
        adj:
            Adjacency matrix of the graph (dim: n_nodes x n_nodes).

        Returns
        ----------
        node_embeddings:
            Output node embeddings (dim: n_nodes x nout).
        """
        x = torch.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x
