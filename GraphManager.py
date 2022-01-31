import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import numpy as np
import enum
import pandas as pd
from DataLoader import DataLoader
from tqdm.notebook import tqdm
from typing import Literal
import pickle



# This is just a Informative class. Changing it will not change anything
class CONSTANTS(enum.Enum):
    node_types = ['team', 'player']
    edge_types = [
        ('team', 'win', 'team'), #DONE
        ('team', 'tie', 'team'), #DONE
        ('team', 'loss', 'team'), #DONE
        ('player', 'playedin', 'team'), #DONE
        ('team', 'used', 'player'), #DONE
        ('player', 'before', 'player'), #DONE
        ('player', 'after', 'player'), #DONE
        ('team', 'before', 'team'), #DONE
        ('team', 'after', 'team') #DONE
    ]


class GraphManager:
    def __init__(self, dl: DataLoader, DEVICE: str):
        self.DEVICE = DEVICE
        self.dl = dl
        self.graph_list = []
        self.train_mask = []
        self.validation_mask = []
        self.test_mask = []
        

    def make(self, df: pd.DataFrame, 
        mode: Literal['CG', 'CW']='CG',
        validation_portion: float=0.1,
        test_portion: float=0.1,
        saveto: str=None
    ):
        self.graph_list = []
        self.train_mask = []
        self.validation_mask = []
        self.test_mask = []
        if mode == 'CG':
            self._gen_continuous_growing(df)
        
        self.test_mask = list(range(int((1-test_portion) * len(self.graph_list)), len(self.graph_list)))
        self.validation_mask = list(range(int((1 - (test_portion+validation_portion)) * len(self.graph_list)), int((1-test_portion) * len(self.graph_list))))
        self.train_mask = list(range(0, int((1 - (test_portion+validation_portion)) * len(self.graph_list))))

        if saveto is not None:
            with open(saveto, 'wb') as pf:
                pickle.dump(self, pf)
    


    def _gen_continuous_growing(self, df: pd.DataFrame):
        try:
            for season, season_df in tqdm(df.groupby('season'), desc='Seasons: '):
                for week, week_df in tqdm(season_df.groupby('week'), desc='Week: ', leave=False):
                    g = self._gen_heterodata(
                        df=df.loc[:week_df.index[-1], :],
                        supervision_indcs=week_df.index.values
                    )
                    self.graph_list.append(g)
        except KeyboardInterrupt:
            pass

    def _gen_heterodata(self, df: pd.DataFrame, messaging_indcs=0, supervision_indcs=0, remove_supervision_links: bool=True):
        hetero_data = pyg_data.HeteroData()
        entity_names = self.dl.DatasetDataframetoNumpy(df)

        #============ Creating Nodes =================
        team_node_features = self.dl.labeler.transform(np.stack((
            entity_names[0],
            entity_names[1]
        )).T.flatten())
        hetero_data['team'].x = torch.tensor(team_node_features, dtype=torch.int, device=self.DEVICE)

        player_node_features = self.dl.labeler.transform(np.moveaxis(np.stack((
            entity_names[2],
            entity_names[3]
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

        # ============ creating used and played_in links ===========
        hetero_data['team', 'used', 'player'].edge_index = torch.stack((
            torch.tensor(np.repeat(team_node_ids.to_numpy(), self.dl.minimum_players_per_team)),
            torch.tensor(player_node_ids.to_numpy())
        )).long().to(self.DEVICE)

        hetero_data['player', 'playedin', 'team'].edge_index = torch.stack((
            torch.tensor(player_node_ids.to_numpy()),
            torch.tensor(np.repeat(team_node_ids.to_numpy(), self.dl.minimum_players_per_team))
        )).long().to(self.DEVICE)


        #=========== Creating Match Result Links ============
        if remove_supervision_links:
            result_df = df.drop(supervision_indcs, axis=0)
        else: result_df = df

        hwins = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'win', :]
            )[0]
        ].to_numpy()

        alosses = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'win', :]
            )[1]
        ].to_numpy()

        hlosses = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'loss', :]
            )[0]
        ].to_numpy()

        awins = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'loss', :]
            )[1]
        ].to_numpy()

        hties = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'tie', :]
            )[0]
        ].to_numpy()

        aties = team_node_ids[
            self.dl.DatasetDataframetoNodeText(
                result_df.loc[result_df['result'] == 'tie', :]
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



        #============ Creating Teams Before Afters ==============
        before_edge_index_list = [[], []]
        for team in np.unique(np.concatenate((entity_names[0], entity_names[1]))):
            team_node_texts = []
            for idx in self.dl.team_play_list[team]:
                if idx > df.index[-1]: break
                if idx not in df.index: continue
                # if idx < df.index[0]: continue
                # elif idx > df.index[-1]: break
                else:
                    team_node_texts.append(f'{team}*{idx}')
            before_edge_index_list[0].extend(team_node_ids[team_node_texts].to_list()[:-1])
            before_edge_index_list[1].extend(team_node_ids[team_node_texts].to_list()[1:])
        
        hetero_data['team', 'before', 'team'].edge_index = torch.tensor(
            before_edge_index_list,
            dtype=torch.long,
            device=self.DEVICE
        )
        hetero_data['team', 'after', 'team'].edge_index = torch.stack((
            torch.tensor(before_edge_index_list[1]),
            torch.tensor(before_edge_index_list[0])
        )).long().to(self.DEVICE)

        
        #============ Creating Players Before After ==============
        before_edge_index_list = [[], []]
        for player in np.unique(np.concatenate((entity_names[2], entity_names[3]))):
            player_node_texts = []
            for idx in self.dl.player_play_list[player]:
                if idx > df.index[-1]: break
                if idx not in df.index: continue
                # if idx < df.index[0]: continue
                else:
                    player_node_texts.append(f'{player}@{idx}')
            before_edge_index_list[0].extend(player_node_ids[player_node_texts].to_list()[:-1])
            before_edge_index_list[1].extend(player_node_ids[player_node_texts].to_list()[1:])
        
        hetero_data['player', 'before', 'player'].edge_index = torch.tensor(
            before_edge_index_list,
            dtype=torch.long,
            device=self.DEVICE
        )
        hetero_data['player', 'after', 'player'].edge_index = torch.stack((
            torch.tensor(before_edge_index_list[1]),
            torch.tensor(before_edge_index_list[0])
        )).long().to(self.DEVICE)


        #========== Creating Supervision Results ===========
        supervision_nodes_text = self.dl.DatasetDataframetoNodeText(df.loc[supervision_indcs, :])
        hetero_data.home_list = team_node_ids[supervision_nodes_text[0]].to_list()
        hetero_data.away_list = team_node_ids[supervision_nodes_text[1]].to_list()
        
        hetero_data.y = torch.tensor(
            self.dl.ConverDatasetResultstoNumber(df.loc[supervision_indcs, :]),
            device=self.DEVICE
        )

        hetero_data.hgoal = torch.tensor(
            self.dl.dataset.loc[supervision_indcs, self.dl.HOME_GOAL_KEY].to_numpy(),
            device= self.DEVICE
        )
        hetero_data.agoal = torch.tensor(
            self.dl.dataset.loc[supervision_indcs, self.dl.AWAY_GOAL_KEY].to_numpy(),
            device= self.DEVICE
        )


        hetero_data.team_node_ids = team_node_ids
        hetero_data.player_node_ids = player_node_ids


        return hetero_data


def load(filename: str):
        with open(filename, 'rb') as pf:
            return pickle.load(pf)