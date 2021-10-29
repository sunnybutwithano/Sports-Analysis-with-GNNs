import pandas as pd
import typing
from Utils import Globals, nodes_gen, extract_players, home_result
import torch
from torch_geometric.data import HeteroData




def home_won_gen(df: pd.DataFrame, full_data_frame=None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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

  if full_data_frame is None:
    full_data_frame = df
  _, team_nodes = nodes_gen(full_data_frame)

  for winner, loser in zip(winning_hashes, losing_hashes):
    winning_nodes.append(team_nodes[winner]) 
    losing_nodes.append(team_nodes[loser])

  won_edges = torch.tensor(
      [
      winning_nodes,
      losing_nodes
      ], 
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  lost_edges = torch.tensor(
      [
      losing_nodes,
      winning_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  return won_edges, lost_edges


def away_won_gen(df: pd.DataFrame, full_data_frame=None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

  if full_data_frame is None:
    full_data_frame = df
  _, team_nodes = nodes_gen(full_data_frame)

  for winner, loser in zip(winning_hashes, losing_hashes):
    winning_nodes.append(team_nodes[winner]) 
    losing_nodes.append(team_nodes[loser])

  won_edges = torch.tensor(
      [
      winning_nodes,
      losing_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  lost_edges = torch.tensor(
      [
      losing_nodes,
      winning_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  )
  
  return won_edges, lost_edges


def tied_gen(df: pd.DataFrame, full_data_frame=None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

  if full_data_frame is None:
    full_data_frame = df
  _, team_nodes = nodes_gen(full_data_frame)

  for home, away in zip(home_hashes, away_hashes):
    home_nodes.append(team_nodes[home]) 
    away_nodes.append(team_nodes[away])

  home_tied_edges = torch.tensor(
      [
      home_nodes,
      away_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  away_tied_edges = torch.tensor(
      [
      away_nodes,
      home_nodes
      ], 
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  return home_tied_edges, away_tied_edges


def played_used_gen(df: pd.DataFrame) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  team_nodes = list()
  player_nodes = list()

  p_nodes, t_nodes = nodes_gen(df)

  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
    home_players, away_players = extract_players(h_lineup, a_lineup)

    for home_player, away_player in zip(home_players, away_players):
      player_nodes.append(p_nodes[f'{home_player}@{index}'])
      team_nodes.append(t_nodes[f'{h_team}*{index}'])
      player_nodes.append(p_nodes[f'{away_player}@{index}'])
      team_nodes.append(t_nodes[f'{a_team}*{index}'])

  played_in_edges = torch.tensor(
      [
       player_nodes,
       team_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  )

  used_edges = torch.tensor(
      [
       team_nodes,
       player_nodes
      ],
      dtype=torch.long,
      device=Globals.DEVICE.value
  ) 

  return played_in_edges, used_edges


def players_before_after_gen(df: pd.DataFrame) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  player_match_hashes = list()

  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      home_players, away_players = extract_players(h_lineup, a_lineup)

      for player in home_players + away_players:
        player_match_hashes.append(f'{player}@{index}')



  sorted_hashes = sorted(
      player_match_hashes,
      key=lambda w: (w.split('@')[0], int(w.split('@')[1]))
  )

  before_nodes = list()
  after_nodes = list()

  player_nodes, _ = nodes_gen(df)

  for index, hash in enumerate(sorted_hashes):
    player, match = hash.split('@')
    before_node = player_nodes[hash]
    try:
      after_node = player_nodes[sorted_hashes[index+1]]
      before_name = player_match_hashes[before_node].split('@')[0]
      after_name = player_match_hashes[after_node].split('@')[0]
      if before_name == after_name:
        before_nodes.append(before_node)
        after_nodes.append(after_node)
    except:
      pass
  before_edges = torch.tensor(
      [
      before_nodes,
      after_nodes
      ], dtype=torch.long,
      device=Globals.DEVICE.value
  )

  after_edges = torch.tensor(
      [
      after_nodes,
      before_nodes
      ], dtype=torch.long,
      device=Globals.DEVICE.value
  )

  return before_edges, after_edges


def teams_before_after_gen(df: pd.DataFrame) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  team_match_hashes = list()

  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      team_match_hashes.append(f'{h_team}*{index}')
      team_match_hashes.append(f'{a_team}*{index}')

  sorted_hashes = sorted(
      team_match_hashes,
      key= lambda w: (w.split('*')[0], int(w.split('*')[1]))
  )

  before_nodes = list()
  after_nodes = list()

  _, team_nodes = nodes_gen(df)

  for index, hash in enumerate(sorted_hashes):
    team, match = hash.split('*')
    before_node = team_nodes[hash]
    try:
      after_node = team_nodes[sorted_hashes[index+1]]
      before_name = team_match_hashes[before_node].split('*')[0]
      after_name = team_match_hashes[after_node].split('*')[0]
      if before_name == after_name:
        before_nodes.append(before_node)
        after_nodes.append(after_node)
    except:
      pass
  before_edges = torch.tensor(
      [
      before_nodes,
      after_nodes
      ], dtype=torch.long,
      device=Globals.DEVICE.value
  )

  after_edges = torch.tensor(
      [
      after_nodes,
      before_nodes
      ], dtype=torch.long,
      device=Globals.DEVICE.value
  )

  return before_edges, after_edges


def complete_graph_gen(df: pd.DataFrame, for_players: bool=True, for_teams: bool=True) -> dict:
  home_won, away_lost = home_won_gen(df)
  away_won, home_lost = away_won_gen(df)
  home_tied, away_tied = tied_gen(df)
  player_played, team_used = played_used_gen(df)

  if for_players:
    player_before, player_after = players_before_after_gen(df)
  if for_teams:
    team_before, team_after = teams_before_after_gen(df)
  won_edge_index = torch.cat(
      (home_won, away_won),
      dim=1
  )
  lost_edge_index = torch.cat(
      (away_lost, home_lost),
      dim=1
  )
  tied_edge_index = torch.cat(
      (home_tied, away_tied),
      dim=1
  )
  edge_index = {
      'won': won_edge_index,
      'lost': lost_edge_index,
      'tied': tied_edge_index,
      'played': player_played,
      'used': team_used,
      'p_after':player_after,
      'p_before': player_before,
      't_after': team_after,
      't_before': team_after
  }   
  return edge_index


def supervision_graph_gen(df : pd.DataFrame, messaging: list, supervision: list, for_players: bool=True, for_teams: bool=True, log_supervision_matches: bool=False) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    #   if log_supervision_matches:
    #     if model.mode == 'train':
    #       mode = 'training'
    #     elif model.mode == 'dev':
    #       mode = 'validating'
    #     elif model.mode == 'test':
    #       mode = 'testing'
    #     logging.info(
    #         f'Messaging on matches ({messaging[0] + 1} -> {messaging[-1] + 1:>5}),\ Model is {mode} on matches ({last_match+2} -> {last_match + 11})'
    #     )
  target_for_nodes = df

  home_won, away_lost = home_won_gen(df.loc[messaging], full_data_frame=target_for_nodes)
  away_won, home_lost = away_won_gen(df.loc[messaging], full_data_frame=target_for_nodes)
  home_tied, away_tied = tied_gen(df.loc[messaging], full_data_frame=target_for_nodes)

  player_played, team_used = played_used_gen(df)

  if for_players:
    player_before, player_after = players_before_after_gen(df)
  if for_teams:
    team_before, team_after = teams_before_after_gen(df)

  won_edge_index = torch.cat(
      (home_won, away_won),
      dim=1
  )
  lost_edge_index = torch.cat(
      (away_lost, home_lost),
      dim=1
  )
  tied_edge_index = torch.cat(
      (home_tied, away_tied),
      dim=1
  )
  edge_index = {
      'won': won_edge_index,
      'lost': lost_edge_index,
      'tied': tied_edge_index,
      'played': player_played,
      'used': team_used,
      'p_after':player_after,
      'p_before': player_before,
      't_after': team_after,
      't_before': team_after
  }  
  return edge_index


def data_gen(df: pd.DataFrame, messaging: list, supervision: list=None, remove_supervision_links: bool=True, for_players: bool=True, for_teams: bool=True, print_edges: bool=False, log_supervision_matches: bool=False) -> HeteroData:
    #   if print_edges:
    #     show_edges(df, edge_index, edge_type)
  if remove_supervision_links:
    edge_index = supervision_graph_gen(
        df,
        messaging=messaging,
        supervision=supervision,
        for_players=for_players,
        for_teams=for_teams,
        log_supervision_matches=log_supervision_matches
    )
    y = torch.tensor(
        df.loc[supervision]['result'].map(home_result).values,
        device=Globals.DEVICE.value
    )

  else:
    if supervision is None:
      supervision = df.index
    if messaging is None:
      messaging = df.index
    edge_index = complete_graph_gen(df, for_players, for_teams)
    y = torch.tensor(
        df.loc[supervision]['result'].map(home_result).values,
        device=Globals.DEVICE.value
    )

  data = HeteroData()
  data['player'].x = torch.unique(edge_index['played'][0]).to(Globals.DEVICE.value).type(torch.int64)
  data['team'].x = torch.unique(edge_index['used'][0]).to(Globals.DEVICE.value).type(torch.int64)
  
  data['team', 'won', 'team'].edge_index = edge_index['won']
  data['team', 'lost_to', 'team'].edge_index = edge_index['lost']
  data['team', 'tied_with', 'team'].edge_index = edge_index['tied']
  data['player', 'played_for', 'team'].edge_index = edge_index['played']
  data['team', 'used', 'player'].edge_index = edge_index['used']
  data['player', 'is_before', 'player'].edge_index = edge_index['p_before']
  data['player', 'is_after', 'player'].edge_index = edge_index['p_after']
  data['team', 'is_before', 'team'].edge_index = edge_index['t_before']
  data['team', 'is_after', 'team'].edge_index = edge_index['t_after']
  data.y = y

  return data


def batch_gen(df: pd.DataFrame, entities: dict, messaging: list=None, supervision: list=None, remove_supervision_links: bool=True, log_supervision_matches: bool=False) -> HeteroData:
  graph = data_gen(
      df,
      messaging=messaging,
      supervision=supervision, 
      remove_supervision_links=remove_supervision_links,
      log_supervision_matches=log_supervision_matches
  )
  
  home_teams = list()
  away_teams = list()

  p_nodes, t_nodes = nodes_gen(df)
  nodes = {**p_nodes, **t_nodes}
  
  if supervision is None:
    supervision = df.index

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
  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.loc[supervision].iterrows():
      home_teams.append(nodes[f'{h_team}*{index}'])
      away_teams.append(nodes[f'{a_team}*{index}'])

  features_player = torch.tensor(
      [indices[i.item()] for i in graph['player'].x],
      device=Globals.DEVICE.value
  )
  features_team = torch.tensor(
      [indices[i.item()] for i in graph['team'].x],
      device=Globals.DEVICE.value
  )

  graph['player'].x = features_player
  graph['team'].x = features_team
  graph.home_list = home_teams
  graph.away_list = away_teams
  
  return graph