import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, GraphConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(self, embedding_dims, conv_dims, fc_dims, dropout: dict, classify: bool=True):
        super(HeteroGNN, self).__init__()
        self.dropout = dropout
        self.classify = classify
        self.embedding_dims = embedding_dims
        self.embedding = nn.Embedding(num_embeddings=embedding_dims[0], embedding_dim=embedding_dims[1])

        # Insted of Embedding to Embedd One HOT
        # self.onehot_lin = nn.Linear(embedding_dims[0], embedding_dims[1])


        self.convs = nn.ModuleList([
            HeteroConv({
                ('team', 'win', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'loss', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'tie', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'playedin', 'team'): GATConv((conv_dims[i], conv_dims[i]), conv_dims[i+1]),
                ('team', 'used', 'player'): GATConv((conv_dims[i], conv_dims[i]), conv_dims[i+1]),
                ('player', 'before', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'after', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'before', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'after', 'team'): GCNConv(conv_dims[i], conv_dims[i+1])
            }, aggr='sum')
        for i in range(len(conv_dims[:-1]))])



        # To Make up for the embedding in the Convs using one hot
        # self.convs.insert(0, 
        #     HeteroConv({
        #         ('team', 'win', 'team'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('team', 'loss', 'team'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('team', 'tie', 'team'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('player', 'playedin', 'team'): GATConv(embedding_dims[0], embedding_dims[1], heads=1),
        #         ('team', 'used', 'player'): GATConv(embedding_dims[0], embedding_dims[1], heads=1),
        #         ('player', 'before', 'player'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('player', 'after', 'player'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('team', 'before', 'team'): GCNConv(embedding_dims[0], embedding_dims[1]),
        #         ('team', 'after', 'team'): GCNConv(embedding_dims[0], embedding_dims[1])
        #     }, aggr='sum')
        # )



        self.fcs = nn.ModuleList([
            nn.Linear(fc_dims[i], fc_dims[i+1]) for i in range(len(fc_dims[:-1]))
        ])

        self.classifier = nn.LogSoftmax(dim=1)

    
    def reset_parameters(self):
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
        # x_dict = {key: F.one_hot(value.long(), num_classes = self.embedding_dims[0]).float() for key, value in x_dict.items()}
        # x_dict = {key: self.onehot_lin(value) for key, value in x_dict.items()}
        # x_dict = {key: F.dropout(value, p=self.dropout['emb'], training=self.training) for key, value in x_dict.items()}


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
        # h = F.dropout(h, p=self.dropout['fc'], training=self.training)
        
        if self.classify:
            h = self.classifier(h)
        
        return h


class BladeChest(nn.Module):
  def __init__(self, feature_size, blade_chest_size, dropout_p=0.5):
    super(BladeChest, self).__init__()
    self.feature_size = feature_size
    self.blade_chest_size = blade_chest_size
    self.dropout_p = dropout_p
    self.chest_transform = nn.Linear(self.feature_size, self.blade_chest_size, bias=False)
    self.chest_bn = nn.BatchNorm1d(self.blade_chest_size)

    self.blade_transform = nn.Linear(self.feature_size, self.blade_chest_size, bias=False)
    self.blade_bn = nn.BatchNorm1d(self.blade_chest_size)

    self.regularizer = nn.Dropout(p=self.dropout_p)
    self.activation = nn.Tanh()

    self.result_transform = nn.Linear(1, 3)
    self.classifier = nn.LogSoftmax(dim=-1)

  def _encode_team(self, team):
    blade = self.blade_transform(team)
    blade = self.blade_bn(blade)
    blade = self.activation(blade)
    blade = self.regularizer(blade)

    chest = self.chest_transform(team)
    chest = self.chest_bn(chest)
    chest = self.activation(chest)
    chest = self.regularizer(chest)

    return blade, chest


  def _matchup(self, home_blade, home_chest, away_blade, away_chest):
    return (home_blade, away_chest).sum(-1) - (away_blade, home_chest).sum(-1)

  def forward(self, home, away):
    home_blade, home_chest = self._encode_team(home)

    away_blade, away_chest = self._encode_team(away)

    matchup_score = self._matchup(home_blade, home_chest, away_blade, away_chest).reshape(-1, 1)

    result = self.result_transform(matchup_score)
    result = self.classifier(result)
    result = self.regularizer(result)

    return result


class HeteroGNN_BladeChest(nn.Module):
    def __init__(self, embedding_dims, conv_dims, bc_dim, dropout: dict, classify: bool=True):
        super(HeteroGNN_BladeChest, self).__init__()
        self.dropout = dropout
        self.classify = classify
        self.embedding_dims = embedding_dims
        self.embedding = nn.Embedding(num_embeddings=embedding_dims[0], embedding_dim=embedding_dims[1])


        self.convs = nn.ModuleList([
            HeteroConv({
                ('team', 'win', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'loss', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'tie', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'playedin', 'team'): GATConv((conv_dims[i], conv_dims[i]), conv_dims[i+1]),
                ('team', 'used', 'player'): GATConv((conv_dims[i], conv_dims[i]), conv_dims[i+1]),
                ('player', 'before', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('player', 'after', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'before', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                ('team', 'after', 'team'): GCNConv(conv_dims[i], conv_dims[i+1])
            }, aggr='sum')
        for i in range(len(conv_dims[:-1]))])

        self.decoder = BladeChest(conv_dims[-1], bc_dim, dropout['fc'])

    
    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        for fc in self.fcs: fc.reset_parameters()
        self.decoder.reset_parameters()
    

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


        #============ Decoder ============
        h = self.decoder(
            x_dict['team'][home_list],
            x_dict['team'][away_list]
        )
        
        return h