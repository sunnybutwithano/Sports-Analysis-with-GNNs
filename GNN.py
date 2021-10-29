import torch
from torch.nn import Module,\
                     ModuleList,\
                     Embedding,\
                     BatchNorm1d,\
                     LogSoftmax,\
                     Softmax,\
                     Linear,\
                     NLLLoss,\
                     CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn.conv import RGCNConv, GINConv, GATConv, HeteroConv, GCNConv
from typing import NoReturn
import typing









#@title HeteroGNN Model { form-width: "10%" }
class HeteroGNN(Module):
  def __init__(self, embedding_dims: tuple, conv_dims: list, fully_connected_dims: list, dropout: dict)-> NoReturn:
    super(HeteroGNN, self).__init__()

    self.mode = None # 'train' or 'test' or 'dev' later 
    self.output_dim = 3 #home_result: win, lose, tie
    self.num_relations = 7 #win/lose/tie/play/use/after/before
    self.dropout = dropout

    #one-hot to latent
    self.embed = Embedding(embedding_dims[0], embedding_dims[1])
    
    conv_list = [
                  HeteroConv(
                      {
                          ('team', 'won', 'team'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('team', 'lost_to', 'team'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('team', 'tied_with', 'team'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('player', 'played_for', 'team'): GATConv(embedding_dims[-1], conv_dims[0], heads=1),
                          ('team', 'used', 'player'): GATConv(embedding_dims[-1], conv_dims[0], heads=1),
                          ('player', 'is_before', 'player'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('player', 'is_after', 'player'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('team', 'is_before', 'team'): GCNConv(embedding_dims[-1], conv_dims[0]),
                          ('team', 'is_after', 'team'): GCNConv(embedding_dims[-1], conv_dims[0])
                      }, aggr='sum'
                  )
                ] + \
                [
                  HeteroConv(
                      {
                          ('team', 'won', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('team', 'lost_to', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('team', 'tied_with', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('player', 'played_for', 'team'): GATConv(conv_dims[i], conv_dims[i+1], heads=1),
                          ('team', 'used', 'player'): GATConv(conv_dims[i], conv_dims[i+1], heads=1),
                          ('player', 'is_before', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('player', 'is_after', 'player'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('team', 'is_before', 'team'): GCNConv(conv_dims[i], conv_dims[i+1]),
                          ('team', 'is_after', 'team'): GCNConv(conv_dims[i], conv_dims[i+1])
                      }, aggr='sum'
                  )
                  for i in range(len(conv_dims[:-1]))
                ]


              

  
    # batch_norm_list = [
    #                      BatchNorm1d(conv_dims[i])
    #                      for i in range(len(conv_dims[:-1]))
    #                   ]

    fully_connected_list =   [
                                Linear(2*conv_dims[-1], fully_connected_dims[0])
                             ] + \
                             [
                                Linear(fully_connected_dims[i], fully_connected_dims[i+1])
                                for i in range(len(fully_connected_dims[:-1]))
                             ] + \
                             [
                                Linear(fully_connected_dims[-1], self.output_dim)
                             ]
    #graph conv layers
    self.conv_layers = ModuleList(conv_list)
    #batch normalization layers

    # self.batch_norm_layers = ModuleList(batch_norm_list)

    #fully connected dense layers
    self.fully_connected_layers = ModuleList(fully_connected_list)

    self.classifier = LogSoftmax(dim=1)
      

  def reset_parameters(self):
      self.embed.reset_parameters()
      for conv in self.conv_layers:
          # for layer in conv:
          #   layer.reset_parameters()
          conv.reset_parameters()
      # for bn in self.batch_norm_layers:
      #     bn.reset_parameters()
      for fc in self.fully_connected_layers:
          fc.reset_parameters()


  def forward(self, data: HeteroData) -> torch.Tensor:
    x_dict = data.x_dict
    home_list = data.home_list
    away_list = data.away_list

    edge_index_dict = data.edge_index_dict
    x_dict = {key: self.embed(x) for key, x in x_dict.items()}
    
    if self.training:
      x_dict = {key: F.dropout(x, p=self.dropout["emb"]) for key, x in x_dict.items()}

    # for conv, bn in zip(self.conv_layers[:-1], self.batch_norm_layers):
    for conv in self.conv_layers[:-1]:
      x_dict = conv(x_dict, edge_index_dict=edge_index_dict)
      x_dict = {key: F.relu(x) for key, x in x_dict.items()}
      if self.training:
        x_dict = {key: F.dropout(x, p=self.dropout["conv"]) for key, x in x_dict.items()}

    x_dict = self.conv_layers[-1](x_dict, edge_index_dict=edge_index_dict)
    if self.training:
      x_dict = {key: F.dropout(x, p=self.dropout["conv"]) for key, x in x_dict.items()}

    ##################################### End of Encoder 
    h = torch.cat(
        (x_dict['team'][home_list], x_dict['team'][away_list]),
        dim=1
    )

    for fc in self.fully_connected_layers[:-1]:
      h = fc(h)
      h = F.relu(h)
      if self.training:
        h = F.dropout(h, p=self.dropout["fc"])

    h = self.fully_connected_layers[-1](h)
    # if self.training:
    #   h = F.dropout(h, p=self.dropout["fc"])

    return self.classifier(h)

    


#@title GNN Model { form-width: "10%" }
class HomoGNN(Module):
  def __init__(self, embedding_dims: tuple, conv_dims: list, fully_connected_dims: list, dropout: dict)-> NoReturn:
    super(HomoGNN, self).__init__()

    self.mode = None # 'train' or 'test' or 'dev' later 
    self.output_dim = 3 #home_result: win, lose, tie
    self.num_relations = 7 #win/lose/tie/play/use/after/before
    self.dropout = dropout

    #one-hot to latent
    self.embed = Embedding(embedding_dims[0], embedding_dims[1])

    conv_list = [
                  RGCNConv(embedding_dims[1], conv_dims[0], self.num_relations)
                ] + \
                [
                  RGCNConv(conv_dims[i], conv_dims[i+1], self.num_relations)
                  for i in range(len(conv_dims[:-1]))
                ]
  
    batch_norm_list = [
                         BatchNorm1d(conv_dims[i])
                         for i in range(len(conv_dims[:-1]))
                      ]

    fully_connected_list =   [
                                Linear(2*conv_dims[-1], fully_connected_dims[0])
                             ] + \
                             [
                                Linear(fully_connected_dims[i], fully_connected_dims[i+1])
                                for i in range(len(fully_connected_dims[:-1]))
                             ] + \
                             [
                                Linear(fully_connected_dims[-1], self.output_dim)
                             ]
    #graph conv layers
    self.conv_layers = ModuleList(conv_list)
    #batch normalization layers
    self.batch_norm_layers = ModuleList(batch_norm_list)
    #fully connected dense layers
    self.fully_connected_layers = ModuleList(fully_connected_list)

    self.classifier = LogSoftmax(dim=1)

    
  def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        for bn in self.batch_norm_layers:
            bn.reset_parameters()
        for fc in self.fully_connected_layers:
            fc.reset_parameters()
          

  def forward(self, x:torch.Tensor, edge_index:torch.Tensor, edge_type:torch.Tensor, home_list:list, away_list:list) -> torch.Tensor:
    x = self.embed(x)
    if self.training:
      x = F.dropout(x, p=self.dropout["emb"])

    for conv, bn in zip(self.conv_layers[:-1], self.batch_norm_layers):
      x = conv(x, edge_index=edge_index, edge_type=edge_type)
      x = bn(x)
      x = F.relu(x)
      if self.training:
        x = F.dropout(x, p=self.dropout["conv"])


    x = self.conv_layers[-1](x, edge_index, edge_type)
    if self.training:
      x = F.dropout(x, p=self.dropout["conv"])

    ##################################### End of Encoder 

    h = torch.cat(
        (x[home_list], x[away_list]),
        dim=1
    )

    for fc in self.fully_connected_layers[:-1]:
      h = fc(h)
      h = F.relu(h)
      if self.training:
        h = F.dropout(h, p=self.dropout["fc"])

    h = self.fully_connected_layers[-1](h)
    if self.training:
      h = F.dropout(h, p=self.dropout["fc"])

    return self.classifier(h)



if __name__ == "__main__":
    hetero_model = HeteroGNN((100,10), [10,10], [10,10], {'emb': 0.2, 'conv': 0.2, 'fc': 0.2})
    normal_model = HomoGNN((100,10), [10,10], [10,10], {'emb': 0.2, 'conv': 0.2, 'fc': 0.2})
    print("Models NO ERROR")
    