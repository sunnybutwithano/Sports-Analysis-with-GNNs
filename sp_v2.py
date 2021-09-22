# enter these commands in CLI to install Pytorch-Geometric
# !pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q git+https://github.com/rusty1s/pytorch_geometric.git


import pandas as pd
import torch
from torch.nn import Module,\
                     ModuleList,\
                     Embedding,\
                     BatchNorm1d,\
                     LogSoftmax,\
                     Linear,\
                     NLLLoss
from torch.optim import Adam
import torch.nn.functional as F
import torch_geometric as PyG
from torch_geometric.data import Data
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.utils import to_networkx
from collections import OrderedDict as od
import logging
import requests
from os import getcwd


# Global Values
WON = 0
LOST_TO = 1
TIED_WITH = 2
PLAYED_IN = 3
USED = 4
BEFORE = 5
AFTER = 6


class GNN(Module):
  def __init__(self, embedding_dims, conv_dims, fully_connected_dims, dropout=0.2):
    super(GNN, self).__init__()

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
                ] + \
                [ 
                 RGCNConv(conv_dims[-1], self.output_dim, self.num_relations)
                ]
  
    batch_norm_list = [
                         BatchNorm1d(conv_dims[i])
                         for i in range(len(conv_dims))
                      ]

    fully_connected_list =   [
                                Linear(2*self.output_dim, fully_connected_dims[0])
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

    self.classifier = LogSoftmax()

    
  def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        for bn in self.batch_norm_layers:
            bn.reset_parameters()


  def forward(self, x, edge_index, edge_type, home_list, away_list):
    x = self.embed(x)

    for conv, bn in zip(self.conv_layers[:-1], self.batch_norm_layers):
      x = conv(x, edge_index=edge_index, edge_type=edge_type)
      x = bn(x)
      x = F.relu(x)
      if self.training:
        x = F.dropout(x, p=self.dropout)

    x = self.conv_layers[-1](x, edge_index, edge_type)

    pred = list()
    for home_team, away_team in zip(home_list, away_list):
      h = torch.cat((x[home_team], x[away_team]))

      for fc in self.fully_connected_layers[:-1]:
        h = fc(h)
        h = F.relu(h)

      h = self.fully_connected_layers[-1](h)
      pred.append(self.classifier(h))

    return torch.stack(pred)


def home_result(row):
  if row == 'home':
    return WON
  elif row == 'tie':
    return TIED_WITH
  elif row == 'away':
    return LOST_TO


def remove_redundancy(players):
  new_players = list()

  for player in players:
    if 'Own' in player:
      player = player.replace('Own', '')
    if 'Pen. Scored' in player:
      player = player.replace('Pen. Scored', '')
    if 'Pen. Score' in player:
      player = player.replace('Pen. Score', '')
    if 'Own' in player or 'Scored' in player or '.' in player or 'Score' in player:
      print(player)
      #SHOULD NOT PRINT IF CODE IS CORRECT
    else:
      new_players.append(player.strip())
  return new_players


def extract_players(home_lineup, away_lineup):
  home_players = home_lineup[:-2].split(' - ')
  away_players = away_lineup[:-2].split(' - ')
  
  return remove_redundancy(home_players), remove_redundancy(away_players)


def stats(df, show_players=False, show_teams=False, show_results=False):
  players_set = set()
  players_list = list()
  teams_set = set()
  teams_list = list()
  results = dict()
  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
    home_players, away_players = extract_players(h_lineup, a_lineup)
    players_set.update(home_players + away_players)
    players_list.extend(home_players + away_players)
    if result == 'home':
      results.update({f'{h_team} #Wins': results.get(f'{h_team} #Wins', 0)+1})
      results.update({f'{a_team} #Losses': results.get(f'{a_team} #Losses', 0)+1})
    elif result == 'tie':
      results.update({f'{h_team} #Ties': results.get(f'{h_team} #Ties', 0)+1})
      results.update({f'{a_team} #Ties': results.get(f'{a_team} #Ties', 0)+1})
    else:
      results.update({f'{a_team} #Wins': results.get(f'{a_team} #Wins', 0)+1})
      results.update({f'{h_team} #Losses': results.get(f'{h_team} #Losses', 0)+1})

    teams_list.extend([h_team, a_team])
    teams_set.update([h_team, a_team])
    
  if show_players:
    for player in players_set:
      print(f'{player} played in {players_list.count(player)} matches.')
  if show_teams:
    for team in teams_set:
      print(f'{team} played {teams_list.count(team)} matches.')
  if show_results:
    results = od(sorted(results.items()))
    for key, val in results.items():
      print(f'{key}: {val}')


def extract_entities(df):
  players_set = set()
  players_list = list()
  teams_set = set()
  teams_list = list()
  # results = dict()
  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
    home_players, away_players = extract_players(h_lineup, a_lineup)

    players_set.update(home_players + away_players)
    teams_set.update([h_team, a_team])
    
  
  return teams_set, players_set


def gen_entities(df):
  teams, players = extract_entities(df)
  entities = {entity: index for index, entity in enumerate(list(players) + list(teams))}
  return entities


def nodes_gen(df):
  nodes = dict()
  node_counter = 0

  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      home_players, away_players = extract_players(h_lineup, a_lineup)

      for player_index, player in enumerate(home_players):
        nodes[f'{player}@{index}'] = node_counter
        node_counter += 1
      for player_index, player in enumerate(away_players):
        nodes[f'{player}@{index}'] = node_counter
        node_counter += 1

      nodes[f'{h_team}*{index}'] = node_counter
      node_counter += 1

      nodes[f'{a_team}*{index}'] = node_counter
      node_counter += 1

  # return od(sorted(nodes.items()))
  return nodes


def show_edges(df, edge, edge_type):
  types = {
      0: 'Won',
      1: 'Lost To',
      2: 'Tied With',
      3: 'Played For',
      4: 'Used As Player',
      5: 'Is Before',
      6: 'Is After'
  }
  nodes = nodes_gen(df)
  r = {k:v for v, k in nodes.items()}
  for i in range(edge_type.shape[0]):
    head = int(edge[0][i].item())
    tail = int(edge[1][i].item())
    relation = int(edge_type[i].item())
    arrow = f'=== {types[relation]} ===>'
    print(f'{r[head]:<32}   {arrow}   {r[tail]:>32}')


def home_won_gen(df):
  home_winning_matches = df.loc[df['result'] == 'home']
  home_winners = home_winning_matches['home_team']
  away_losers = home_winning_matches['away_team']

  winning_hashes = list()
  losing_hashes = list()

  for home, away, match in zip(home_winners, away_losers, home_winners.index):
    winning_hashes.append(f'{home}*{match}')
    losing_hashes.append(f'{away}*{match}')

  winning_nodes = list()
  losing_nodes = list()

  nodes = nodes_gen(df)

  for winner, loser in zip(winning_hashes, losing_hashes):
    winning_nodes.append(nodes[winner]) 
    losing_nodes.append(nodes[loser])

  won_edges = torch.tensor(
      [
      winning_nodes,
      losing_nodes
      ], 
      dtype=torch.long
  )

  lost_edges = torch.tensor(
      [
      losing_nodes,
      winning_nodes
      ],
      dtype=torch.long
  )

  won_edge_types = torch.ones(won_edges.shape[1], dtype=torch.long) * WON
  lost_edge_types = torch.ones(lost_edges.shape[1], dtype=torch.long) * LOST_TO 

  return won_edges, won_edge_types, lost_edges, lost_edge_types


def away_won_gen(df):
  away_winning_matches = df.loc[df['result'] == 'away']
  away_winners = away_winning_matches['away_team']
  home_losers = away_winning_matches['home_team']

  winning_hashes = list()
  losing_hashes = list()

  for home, away, match in zip(home_losers, away_winners, away_winners.index):
    winning_hashes.append(f'{away}*{match}')
    losing_hashes.append(f'{home}*{match}')

  winning_nodes = list()
  losing_nodes = list()

  nodes = nodes_gen(df)

  for winner, loser in zip(winning_hashes, losing_hashes):
    winning_nodes.append(nodes[winner]) 
    losing_nodes.append(nodes[loser])

  won_edges = torch.tensor(
      [
      winning_nodes,
      losing_nodes
      ],
      dtype=torch.long
  )

  lost_edges = torch.tensor(
      [
      losing_nodes,
      winning_nodes
      ],
      dtype=torch.long
  )
  
  won_edge_types = torch.ones(won_edges.shape[1], dtype=torch.long) * WON
  lost_edge_types = torch.ones(lost_edges.shape[1], dtype=torch.long) * LOST_TO 
  
  return won_edges, won_edge_types, lost_edges, lost_edge_types


def tied_gen(df):
  tied_matches = df.loc[df['result'] == 'tie']
  home_teams = tied_matches['home_team']
  away_teams = tied_matches['away_team']

  home_hashes = list()
  away_hashes = list()

  for home, away, match in zip(home_teams, away_teams, away_teams.index):
    away_hashes.append(f'{away}*{match}')
    home_hashes.append(f'{home}*{match}')

  home_nodes = list()
  away_nodes = list()

  nodes = nodes_gen(df)

  for home, away in zip(home_hashes, away_hashes):
    home_nodes.append(nodes[home]) 
    away_nodes.append(nodes[away])

  home_tied_edges = torch.tensor(
      [
      home_nodes,
      away_nodes
      ],
      dtype=torch.long
  )

  away_tied_edges = torch.tensor(
      [
      away_nodes,
      home_nodes
      ], 
      dtype=torch.long
  )

  home_tied_edge_types = torch.ones(home_tied_edges.shape[1], dtype=torch.long) * TIED_WITH
  away_tied_edge_types = torch.ones(away_tied_edges.shape[1], dtype=torch.long) * TIED_WITH

  return home_tied_edges, home_tied_edge_types, away_tied_edges, away_tied_edge_types


def played_used_gen(df):
  team_nodes = list()
  player_nodes = list()

  nodes = nodes_gen(df)

  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
    home_players, away_players = extract_players(h_lineup, a_lineup)

    for home_player, away_player in zip(home_players, away_players):
      player_nodes.append(nodes[f'{home_player}@{index}'])
      team_nodes.append(nodes[f'{h_team}*{index}'])
      player_nodes.append(nodes[f'{away_player}@{index}'])
      team_nodes.append(nodes[f'{a_team}*{index}'])

  played_in_edges = torch.tensor(
      [
       player_nodes,
       team_nodes
      ],
      dtype=torch.long
  )

  played_in_edge_types = torch.ones(played_in_edges.shape[1], dtype=torch.long) * PLAYED_IN

  used_edges = torch.tensor(
      [
       team_nodes,
       player_nodes
      ],
      dtype=torch.long
  ) 

  used_edge_types = torch.ones(used_edges.shape[1], dtype=torch.long) * USED

  return played_in_edges, played_in_edge_types, used_edges, used_edge_types


#TODO
def players_before_after_gen(df):
  player_match_hashes = list()

  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      home_players, away_players = extract_players(h_lineup, a_lineup)

      for player in home_players + away_players:
        player_match_hashes.append(f'{player}@{index}')

  sorted_hashes = sorted(
      player_match_hashes,
      key= lambda w: (w.split('@')[0], int(w.split('@')[1]))
  )

  before_nodes = list()
  after_nodes = list()

  nodes = nodes_gen(df)

  for index, hash in enumerate(sorted_hashes):
    player, match = hash.split('@')
    before_node = nodes[hash]
    try:
      after_node = nodes[sorted_hashes[index+1]]
      before_name = hashes[before_node].split('@')[0]
      after_name = hashes[after_node].split('@')[0]
      if before_name == after_name:
        before_nodes.append(before_node)
        after_nodes.append(after_node)
    except:
      pass
  before_edges = torch.tensor(
      [
      before_nodes,
      after_nodes
      ], dtype=torch.long
  )

  before_edge_types = torch.ones(before_edges.shape[1], dtype=torch.long) * BEFORE

  after_edges = torch.tensor(
      [
      after_nodes,
      before_nodes
      ], dtype=torch.long
  )

  after_edge_types = torch.ones(after_edges.shape[1], dtype= torch.long) * AFTER

  return before_edges, before_edge_types, after_edges, after_edge_types


def teams_before_after_gen(df):
  team_match_hashes = list()

  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      team_match_hashes.append(f'{h_team}*{index}')
      team_match_hashes.append(f'{a_team}*{index}')

  sorted_hashes = sorted(
      team_match_hashes,
      key= lambda w: (w.split('*')[0], int(w.split('*')[1]))
  )

  before_nodes = list()
  after_nodes = list()

  nodes = nodes_gen(df)

  for index, hash in enumerate(sorted_hashes):
    team, match = hash.split('*')
    before_node = nodes[hash]
    try:
      after_node = nodes[sorted_hashes[index+1]]
      before_name = hashes[before_node].split('*')[0]
      after_name = hashes[after_node].split('*')[0]
      if before_name == after_name:
        before_nodes.append(before_node)
        after_nodes.append(after_node)
    except:
      pass
  before_edges = torch.tensor(
      [
      before_nodes,
      after_nodes
      ], dtype=torch.long
  )

  before_edge_types = torch.ones(before_edges.shape[1], dtype=torch.long) * BEFORE

  after_edges = torch.tensor(
      [
      after_nodes,
      before_nodes
      ], dtype=torch.long
  )

  after_edge_types = torch.ones(after_edges.shape[1], dtype=torch.long) * AFTER

  return before_edges, before_edge_types, after_edges, after_edge_types


def complete_graph_edge_gen(df, for_players=True, for_teams=True):
  home_win, won1, away_lost, lost1 = home_won_gen(df)
  away_won, won2, home_lost, lost2 = away_won_gen(df)
  home_tied, tied1, away_tied, tied2 = tied_gen(df)
  player_played, played1, team_used, used1 = played_used_gen(df)

  edge_index = torch.cat(
        (home_win, away_lost, away_won, home_lost, home_tied, away_tied, player_played, team_used),
        dim=1
    )
  
  edge_type = torch.cat(
        (won1, lost1, won2, lost2, tied1, tied2, played1, used1)
    )

  if for_players:
    player_before, before1, player_after, after1 = players_before_after_gen(df)
    edge_index = torch.cat((edge_index, player_before, player_after), dim=1)
    edge_type = torch.cat((edge_type, before1, after1))
  if for_teams:
    team_before, before2, team_after, after2 = teams_before_after_gen(df)
    edge_index = torch.cat((edge_index, team_before, team_after), dim=1)
    edge_type = torch.cat((edge_type, before2, after2))

  return edge_index, edge_type


def supervision_graph_gen(df, for_players=True, for_teams=True, log_supervision_matches=False):
  if df.shape[0] > 10:
    first_match = df.index[0]
    last_match = df.index[-11]
  else:
    first_match = df.index[1]
    last_match = df.index[df.shape[0] * -1]
  if log_supervision_matches:
    logging.info(f'Messaging on matches ({first_match + 1} -> {last_match + 1}), Model is {"training on" if model.training else "evaluating on"} matches ({last_match+2} -> {last_match + 11})')
    
  home_win, won1, away_lost, lost1 = home_won_gen(df.loc[first_match: last_match, :])
  away_won, won2, home_lost, lost2 = away_won_gen(df.loc[first_match: last_match, :])
  home_tied, tied1, away_tied, tied2 = tied_gen(df.loc[first_match: last_match, :])
  player_played, played1, team_used, used1 = played_used_gen(df)

  edge_index = torch.cat(
        (home_win, away_lost, away_won, home_lost, home_tied, away_tied, player_played, team_used),
        dim=1
    )
  
  edge_type = torch.cat(
        (won1, lost1, won2, lost2, tied1, tied2, played1, used1)
    )
  
  if for_players:
    player_before, before1, player_after, after1 = players_before_after_gen(df)
    edge_index = torch.cat((edge_index, player_before, player_after), dim=1)
    edge_type = torch.cat((edge_type, before1, after1))
  if for_teams:
    team_before, before2, team_after, after2 = teams_before_after_gen(df)
    edge_index = torch.cat((edge_index, team_before, team_after), dim=1)
    edge_type = torch.cat((edge_type, before2, after2))

  return edge_index, edge_type


def data_gen(df, remove_supervision_links=True, for_players=True, for_teams=True, print_edges=False, log_supervision_matches=False):
  if print_edges:
    show_edges(df, edge_index, edge_type)
  if remove_supervision_links:
    edge_index, edge_type = supervision_graph_gen(df, for_players=for_players, for_teams=for_teams, log_supervision_matches=log_supervision_matches)
    if df.shape[0] > 10:
      first_supervision_match = df.index[-10]
      last_supervision_match = df.index[-1]
    else:
      first_supervision_match = df.index[0]
      last_supervision_match = df.index[-1]
    y = torch.tensor(df.loc[first_supervision_match:last_supervision_match, :]['result'].map(home_result).values)

  else:
    edge_index, edge_type = complete_graph_gen(df, for_players, for_teams)
    y = torch.tensor(df['result'].map(home_result).values)

  x = torch.tensor(torch.unique(edge_index), dtype=torch.int64)
  
  data = Data(x=x, y=y, edge_index=edge_index, edge_type = edge_type)
  return data


def visualize_graph(df, width=20, height=20, title=None, remove_supervision_links=False):
  import networkx as nx
  import matplotlib.pyplot as plt
  nodes = nodes_gen(df)
  r = {k:v for v, k in nodes.items()}
  d = data_gen(df, remove_supervision_links=remove_supervision_links)
  G = to_networkx(d)
  types = {
        0: 'Won',
        1: 'Lost To',
        2: 'Tied With',
        3: 'Played For',
        4: 'Used As Player',
        5: 'Is Before',
        6: 'Is After'
  }

  type_color = {
      0: '#00ff00', #won
      1: '#ff0000', #lost to
      2: '#e6d70e', #tied with
      3: '#1338f0', #played for
      4: '#f01373', #used as player
      5: '#0f072e', #is before
      6: '#d909cb' #is after
  }

  double_edge_types = {
      0: '(Won[green] - Lost to[red])',
      1: '(Lost to[red] - Won[green])',
      2: '(Tied with[yellow])',
      3: '(Played for[blue] - Used as Player[pink])',
      4: '(Used as Player[pink] - Played for[blue])',
      5: '(Is Before[dark blue] - Is After[purple])',
      6: '(Is After[purple] - Is Before[dark blue])'
  }

  link_colors = dict(zip(
        types.values(),
        type_color.values()
      )
  )

  node_colors = {
      'player-color': '#8f0ba1',
      'team-color': '#02fae1'   
  }

  all_colors = link_colors.copy()
  all_colors.update(node_colors)

  

  for color_use in all_colors.keys():
      plt.scatter([],[], c=[all_colors[color_use]], label=f'{color_use}')

  edge_colors = list()
  edge_labels = dict()

  for edge in G.edges():
    e = torch.tensor(edge)
    for index, node_node in enumerate(d.edge_index.t()):
      if torch.equal(e, node_node):
        edge_colors.append(type_color[d.edge_type[index].item()])
        label = double_edge_types[d.edge_type[index].item()]
        edge_labels.update({edge:label})
  colors = list()
  node_labels = dict()
  for node in G.nodes():
    if '@' in r[node]:
      colors.append(all_colors['player-color'])
      node_labels.update({node: r[node].split('@')[0]})
    elif '*' in r[node]:
      colors.append(all_colors['team-color'])
      node_labels.update({node:r[node].split('*')[0]})

  fig = plt.gcf()
  fig.set_size_inches(width, height)
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, node_color=colors)
  nx.draw_networkx_labels(G, pos, labels=node_labels)
  nx.draw_networkx_edges(G, pos, edge_color=edge_colors, connectionstyle='arc3,rad=0.05')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
  plt.legend()
  plt.title(title)
  fig.show()
  plt.show()


def batch_gen(df, entities, remove_supervision_links=True, log_supervision_matches=False):
  graph = data_gen(df, remove_supervision_links=remove_supervision_links, log_supervision_matches=log_supervision_matches)
  
  home_teams = list()
  away_teams = list()

  nodes = nodes_gen(df)
  
  indices = dict()
  for hash, index in nodes.items():
    if '@' in hash:
      player = hash.split('@')[0]
      player_id = entities[player]
      indices.update({index:player_id})
    elif '*' in hash:
      team = hash.split('*')[0]
      team_id = entities[team]
      indices.update({index: team_id})
  for index, (h_team, a_team, result, h_lineup, a_lineup) in df.loc[df.index[-1 * graph.y.shape[0]]:, :].iterrows():
      home_teams.append(nodes[f'{h_team}*{index}'])
      away_teams.append(nodes[f'{a_team}*{index}'])

  
  features = torch.tensor(
      [indices[i.item()] for i in graph.x]
  )
  graph_data = {
      "x": features,
      "edge_index": graph.edge_index,
      "edge_type": graph.edge_type,
      "home_teams": home_teams,
      "away_teams": away_teams,
      "y": graph.y
  }

  return graph_data


def train(model, graph_data, optimizer, loss_fn):
  batch_loss = 0

  # model.train()

  out = model(
      x=graph_data["x"],
      edge_index=graph_data["edge_index"],
      edge_type=graph_data["edge_type"],
      home_list=graph_data["home_teams"],
      away_list=graph_data["away_teams"]
  )

  optimizer.zero_grad()
  loss = loss_fn(out, graph_data["y"])
  batch_loss = loss.item()
  loss.backward()
  optimizer.step()

  return batch_loss


@torch.no_grad()
def evaluate(model, graph_data=4):
  all = 0
  correct = 0

  # model.eval()
  out = model(
      x=graph_data["x"],
      edge_index=graph_data["edge_index"],
      edge_type=graph_data["edge_type"],
      home_list=graph_data["home_teams"],
      away_list=graph_data["away_teams"]
  )

  prediction = out.argmax(dim=-1)
  correct = torch.tensor((prediction == graph_data["y"]), dtype=torch.int).sum().item()
  all = graph_data["y"].shape[0]
  # model.train()

  return correct, all


if __name__ == '__main__':
  url = "https://raw.githubusercontent.com/jokecamp/FootballData/master/EPL%202011-2019/PL_scraped_ord.csv"
  # current_directory = getcwd()
  filename = 'dataset.txt'
  req = requests.get(url)
  if req.status_code == 200:
    with open(filename, 'wb') as fp:
      fp.write(req.content)

  #@title Dataset Loading and Cleaning { form-width: "15px" }
  dataset = pd.read_csv(filename, encoding='latin-1', usecols=['home_team', 'away_team', 'result', 'home_lineup', 'away_lineup'])
  corrupted = dataset.loc[pd.isna(dataset['away_lineup']) | pd.isna(dataset['home_lineup'])]
  dataset = dataset.drop(corrupted.index, axis=0)

  logging.basicConfig(
      filename='model-logs.log',
      filemode='w',
      level=logging.INFO
  )

  log_supervision_matches = True
  learning_rate = 1e-3
  num_epochs = 100
  dropout = 0.5
  remove_supervision_links = True
  entities = gen_entities(dataset)

  ######################################## Scheme 3
  train_messaging_graph_size = 440
  val_messaging_graph_size = 440
  test_messaging_graph_size = 440
  iter_size = 10
  val_week_denom = 50
  test_week_denom = 60
  ######################################## Parameters

  model = GNN(
      embedding_dims=(max(entities.values()) + 1, 64),
      conv_dims=(32, 16, 8),
      fully_connected_dims=(24, 24),
      dropout=dropout
  )
  optimizer = Adam(
      model.parameters(),
      lr=learning_rate
  )
  criterion = NLLLoss()

  try:
    
    for epoch in range(num_epochs):
      epoch_loss = 0
      val_correct = 0
      val_all = 0

      for i in range(train_messaging_graph_size, dataset.shape[0], iter_size):
        
        if i % val_week_denom == 0:
          ######################## Validation ########################
          model.eval()

          from_match = i - val_messaging_graph_size
          to_match = i - 1

          validation_df = dataset.loc[from_match: to_match, :]
          val_graph_data = batch_gen(
                validation_df,
                entities=entities,
                remove_supervision_links=remove_supervision_links,
                log_supervision_matches=log_supervision_matches
            )

          val_batch_correct, val_batch_all = evaluate(
              model=model,
              graph_data=val_graph_data
          )

          val_correct += val_batch_correct
          val_all += val_batch_all

        elif i % test_week_denom == 0:
          ######################## Test ########################
          # print(f'Test Week')
          pass

        else:
          ######################## Train ########################
          model.train()

          from_match = i - train_messaging_graph_size
          to_match = i - 1

          train_df = dataset.loc[from_match: to_match, :]
          train_graph_data = batch_gen(
              train_df,
              entities=entities,
              remove_supervision_links=remove_supervision_links,
              log_supervision_matches=log_supervision_matches
          )
          epoch_loss += train(
                model=model,
                graph_data=train_graph_data,
                optimizer=optimizer,
                loss_fn=criterion
            )


      ########## end of epoch ###########
      print(f'{"="*32} Epoch {epoch + 1} {"="*32}')
      print(f'Train Loss:          {epoch_loss:.4f}')
      print(f'Validation Accuracy: {val_correct * 100 / val_all:.3f}%')
      logging.info(f'{"="*32} Epoch {epoch + 1} {"="*32}')
      logging.info(f'Train Loss:          {epoch_loss:.4f}')
      logging.info(f'Validation Accuracy: {val_correct * 100 / val_all:.3f}%')

  except KeyboardInterrupt:
    pass

  test_correct = 0
  test_all = 0
  for i in range(train_messaging_graph_size, dataset.shape[0], iter_size):
    if i % test_week_denom == 0:
      model.eval()
      ######################## Test ########################
      from_match = i - test_messaging_graph_size
      to_match = i - 1

      test_df = dataset.loc[from_match: to_match, :]
      test_graph_data = batch_gen(
          test_df,
          entities=entities,
          remove_supervision_links=remove_supervision_links,
          log_supervision_matches=log_supervision_matches
      )
      test_batch_correct, test_batch_all = evaluate(
          model=model,
          graph_data=test_graph_data
      )
      test_correct += test_batch_correct
      test_all += test_batch_all

  print(f'Test Accuracy: {test_correct * 100 / test_all:.3f}%')
  logging.info('=' * 70)
  logging.info(f'Test Accuracy: {test_correct * 100 / test_all:.3f}%')

