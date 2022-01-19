import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import numpy as np
import enum
import pandas as pd
from DataLoader import DataLoader
from typing import Tuple



# This is just a Informative class. Changing it will not change anything
class CONSTANTS(enum.Enum):
    node_types = ['team', 'player']
    edge_types = [
        ('team', 'win', 'team'), #DONE
        ('team', 'tie', 'team'), #DONE
        ('team', 'loss', 'team'), #DONE
        ('player', 'playedin', 'team'), #DONE
        ('team', 'used', 'player'), #DONE
        ('player', 'before', 'player'),
        ('player', 'after', 'player'),
        ('team', 'before', 'team'),
        ('team', 'after', 'team')
    ]


class GraphManager:
    def __init__(self, dl: DataLoader, dataset: pd.DataFrame, DEVICE: str):
        self.DEVICE = DEVICE
        self.dl = dl
        self.dataset = dataset
        self.graph_list = []
        self.train_mask = []
        self.validation_mask = []
        self.test_mask = []
        

    def _DataLoaderNodeTexttoNodeID(nodetext: Tuple[np.ndarray]):
        team_node_to_id = dict()
        team_id_to_node = dict()
        player_node_to_id = dict()
        player_id_to_node = dict()
        node_id = 0

        for hteam, ateam in zip(nodetext[0], nodetext[1]):
            team_node_to_id[hteam] = node_id
            team_id_to_node[node_id] = hteam
            node_id += 1
            team_node_to_id[ateam] = node_id
            team_id_to_node[node_id] = hteam
            node_id +=1

        node_id = 0
        
        for hlineup, alineup in zip(nodetext[2], nodetext[3]):
            for p in hlineup:
                player_node_to_id[p] = node_id
                player_id_to_node[node_id] = p
                node_id +=1
            for p in alineup:
                player_node_to_id[p] = node_id
                player_id_to_node[node_id] = p
                node_id +=1

        return team_node_to_id, team_id_to_node, player_node_to_id, player_id_to_node




    def _gen_heterodata(self, df: pd.DataFrame, messaging_indcs=0, supervision_indcs=0, remove_supervision_links: bool=True):
        hetero_data = pyg_data.HeteroData()

        #============ Creating Nodes =================
        team_node_features = self.dl.labeler.transform(np.stack((
            self.dl.DatasetDataframetoNumpy(df)[0],
            self.dl.DatasetDataframetoNumpy(df)[1]
        )).T.flatten())
        hetero_data['team'].x = torch.tensor(team_node_features, dtype=torch.int, device=self.DEVICE)

        player_node_features = self.dl.labeler.transform(np.moveaxis(np.stack((
            self.dl.DatasetDataframetoNumpy(df)[2],
            self.dl.DatasetDataframetoNumpy(df)[3]
        )), 0, 1).flatten())
        hetero_data['player'].x = torch.tensor(player_node_features, dtype=torch.int, device=self.DEVICE)



        #========= Creating Node IDs to use in edge index ==========
        node_texts = self.dl.DatasetDataframetoNodeText(df)
        team_node_ids = pd.Series(
            np.arange(node_texts[0].shape[0] * 2),
            index=np.stack((
                node_texts[0],
                node_texts[1]
            )).T.flatten()
        )
        player_node_ids = pd.Series(
            np.arange(node_texts[2].shape[0] * node_texts[2].shape[1] * 2),
            index=np.moveaxis(np.stack((
                node_texts[2],
                node_texts[3]
            )), 0, 1).flatten()
        )

        #============ creating used and played_in links ===========
        hetero_data['team', 'used', 'player'].edge_index = torch.stack((
            torch.tensor(np.repeat(team_node_ids.to_numpy(), self.dl.minimum_players_per_team)),
            torch.tensor(player_node_ids.to_numpy())
        )).long().to(self.DEVICE)

        hetero_data['player', 'playedin', 'team'].edge_index = torch.stack((
            torch.tensor(player_node_ids.to_numpy()),
            torch.tensor(np.repeat(team_node_ids.to_numpy(), self.dl.minimum_players_per_team))
        )).long().to(self.DEVICE)


        #=========== Creating Match Result Links ============
        hwins = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'win', :]
            )[0]
        ].to_numpy()

        alosses = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'win', :]
            )[1]
        ].to_numpy()

        hlosses = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'loss', :]
            )[0]
        ].to_numpy()

        awins = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'loss', :]
            )[1]
        ].to_numpy()

        hties = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'tie', :]
            )[0]
        ].to_numpy()

        aties = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                df.loc[df['result'] == 'tie', :]
            )[1]
        ].to_numpy()

        hetero_data['team', 'win', 'team'].edge_index = torch.stack((
            torch.tensor(np.concatenate((hwins, awins))),
            torch.tensor(np.concatenate((alosses, hlosses)))
        )).long().to(self.DEVICE)

        hetero_data['team', 'loss', 'team'].edge_index = torch.stack((
            torch.tensor(np.concatenate((alosses, hlosses))),
            torch.tensor(np.concatenate((hwins, awins)))
        )).long().to(self.DEVICE)

        hetero_data['team', 'tie', 'team'].edge_index = torch.stack((
            torch.tensor(np.concatenate((hties, aties))),
            torch.tensor(np.concatenate((aties, hties)))
        )).long().to(self.DEVICE)



        return hetero_data

        for idx, match in df.iterrows():
            hteam = match[self.dl.HOME_TEAM_KEY]
            ateam = match[self.dl.AWAY_TEAM_KEY]
            hlineup = match[self.dl.HOME_LINEUP_KEY]
            alineup = match[self.dl.AWAY_LINEUP_KEY]

