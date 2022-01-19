import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from torch_geometric.nn import HeteroConv, GCNConv, GATConv


class HeteroGNN(nn.Module):
    def __init__(self, embedding_dims, conv_dims, fc_dims, dropout: dict):
        super(HeteroGNN, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=embedding_dims[0], embedding_dim=embedding_dims[1])

        self.convs = nn.ModuleList([
            HeteroConv({
                ('team', 'win', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'loss', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'tie', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'playedin', 'team'): GATConv(conv_dims[i], conv_dims[i+1], heads=1),
                ('team', 'used', 'player'): GATConv(conv_dims[i], conv_dims[i+1], heads=1),
                ('player', 'before', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'after', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'before', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'after', 'team'): GCNConv(conv_dims[i], conv_dims[i+1])
            }, aggr='sum')
        for i in range(len(conv_dims[:-1]))])

        self.fcs = nn.ModuleList([
            nn.Linear(fc_dims[i], fc_dims[i+1]) for i in range(len(fc_dims[:-1]))
        ])

        self.classifier = nn.LogSoftmax(dim=1)

    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for conv in self.convs: conv.reset_parameters()
        for fc in self.fcs: fc.reset_parameters()
    

    def forward(self, g: pyg_data.HeteroData):
        x_dict: dict = g.x_dict
        edge_index_dict: dict = g.edge_index_dict
        home_list = g.home_list
        away_list = g.away_list


        #============= Embedding ===========
        x_dict = {key: self.embedding(value) for key, value in x_dict.items()}
        x_dict = {key: F.dropout(value, p=self.dropout['emb'], training=self.training) for key, value in x_dict.items()}


        #============ Convolution ============
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}
            x_dict = {key: F.dropout(value, p=self.dropout['conv'], training=self.training) for key, value in x_dict.items()}
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        x_dict = {key: F.dropout(value, p=self.dropout['conv'], training=self.training) for key, value in x_dict.items()}


        #============ Fully Connected ============
        h = torch.cat((
            x_dict['team'][home_list],
            x_dict['team'][away_list]
        ), dim=1)

        for fc in self.fcs[:-1]:
            h = fc(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout['fc'], training=self.training)
        h = self.fcs[-1](h)
        
        
        return self.classifier(h)

