import torch.nn as nn
from torch.nn import LayerNorm
from module.layers import GATRConv, get_batch_geometric


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_relations):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_relations = num_relations
        self.heads = 4

        self.gnns = nn.ModuleList(
            GATRConv(self.in_channels, self.out_channels, num_relations=self.num_relations, heads=self.heads, concat=False)
            for _ in range(self.num_layers))
        self.rnn = nn.GRUCell(self.out_channels, self.out_channels)

        self.layer_norm = LayerNorm(self.out_channels)

    def forward(self, emb_lst, graphs):
        # emb_lst: list of num_nodes x dim
        # graphs: list of (edges_index, edges_type)
        batch_geometric = get_batch_geometric(emb_lst, graphs)
        memory_bank = batch_geometric.x

        for layer in self.gnns:
            new_memory_bank = layer(memory_bank, batch_geometric.edge_index, edge_type=batch_geometric.y)
            memory_bank = self.rnn(new_memory_bank, memory_bank)

        out = self.layer_norm(memory_bank)

        return out
