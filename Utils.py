import enum
import torch
import typing
import pandas as pd


class Globals(enum.Enum):
    WON = 0
    LOST_TO = 1
    TIED_WITH = 2
    PLAYED_IN = 3
    USED = 4
    BEFORE = 5
    AFTER = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def home_result(result: str) -> int:
  if result == 'win':
    return Globals.WON.value
  elif result == 'tie':
    return Globals.TIED_WITH.value
  elif result == 'loss':
    return Globals.LOST_TO.value


def remove_redundancy(players: list) -> list:
  new_players = list()

  for player in players:
    if 'Own' in player:
      player = player.replace('Own', '')
    if 'Pen. Scored' in player:
      player = player.replace('Pen. Scored', '')
    if 'Pen. Score' in player:
      player = player.replace('Pen. Score', '')
    if 'Own' in player or 'Scored' in player or 'Score' in player:
      print(player)
      #SHOULD NOT PRINT IF CODE IS CORRECT
    else:
      new_players.append(player.strip())
  return new_players


def extract_players(home_lineup: str, away_lineup: str, seperator: str=' - ') -> list:
  home_players = home_lineup.split(seperator)
  away_players = away_lineup.split(seperator)
  
  return remove_redundancy(home_players), remove_redundancy(away_players)


def extract_entities(df: pd.DataFrame) -> typing.Tuple[set, set]:
  players_set = set()
  players_list = list()
  teams_set = set()

  teams_list = list()
  # results = dict()
  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
    home_players, away_players = extract_players(h_lineup, a_lineup)

    players_set.update(home_players + away_players)
    teams_set.update([h_team, a_team])
    
  return teams_set, players_set


def gen_entities(df: pd.DataFrame) -> dict:
  teams, players = extract_entities(df)
  entities = {entity: index for index, entity in enumerate(list(players) + list(teams))}
  return entities


def nodes_gen(df: pd.DataFrame) -> typing.Tuple[dict, dict]:
  player_nodes = dict()
  team_nodes = dict()
  player_node_counter = 0
  team_node_counter = 0

  for index, (league, season, week, h_team, a_team, result, h_lineup, a_lineup) in df.iterrows():
      home_players, away_players = extract_players(h_lineup, a_lineup)

      for player_index, player in enumerate(home_players):
        player_nodes[f'{player}@{index}'] = player_node_counter
        player_node_counter += 1
      for player_index, player in enumerate(away_players):
        player_nodes[f'{player}@{index}'] = player_node_counter
        player_node_counter += 1

      team_nodes[f'{h_team}*{index}'] = team_node_counter
      team_node_counter += 1

      team_nodes[f'{a_team}*{index}'] = team_node_counter
      team_node_counter += 1

  return player_nodes, team_nodes


