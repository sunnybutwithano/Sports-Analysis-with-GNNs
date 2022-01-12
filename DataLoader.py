import pandas as pd
import numpy as np
import sklearn.preprocessing as skp


class DataLoader:
    def __init__(self, filename: str, minimum_players_per_team: int):
        self.filename = filename
        self.minimum_players_per_team = minimum_players_per_team


        #CONSTANTS
        self.DATA_COLUMNS = ['league', 'season', 'week', 'home_team', 'away_team', 'result', 'home_lineup', 'away_lineup']
        self.HOME_TEAM_KEY = 'home_team'
        self.AWAY_TEAM_KEY = 'away_team'
        self.HOME_LINEUP_KEY = 'home_lineup'
        self.AWAY_LINEUP_KEY = 'away_lineup'


        #Reading the file into a DataFrame
        self.dataset: pd.DataFrame = pd.read_csv(
            self.filename,
            encoding='utf-8',
            usecols=self.DATA_COLUMNS,
            dtype=dict(zip(self.DATA_COLUMNS, [str]*2 + [int] + [str]*5))
        )

        self._data_cleanup()





    
    def _data_cleanup(self):
        #Some Names may have Unwanted Characters. Striping them
        self.dataset.loc[:, self.HOME_TEAM_KEY] = self.dataset.loc[:, self.HOME_TEAM_KEY].apply(lambda z: z.strip('123456789., '))
        self.dataset.loc[:, self.AWAY_TEAM_KEY] = self.dataset.loc[:, self.AWAY_TEAM_KEY].apply(lambda z: z.strip('123456789., '))

        #Spliting lineups into lists of players
        self.dataset.loc[:, self.HOME_LINEUP_KEY] = self.dataset.loc[:, self.HOME_LINEUP_KEY].apply(lambda z: z.split(' - '))
        self.dataset.loc[:, self.AWAY_LINEUP_KEY] = self.dataset.loc[:, self.AWAY_LINEUP_KEY].apply(lambda z: z.split(' - '))

        #Droping Matches with Less than Minimum Number of Players
        corrupted = self.dataset[
            self.dataset[[self.HOME_LINEUP_KEY, self.AWAY_LINEUP_KEY]].apply(
                lambda z: len(set(z[0] + z[1])), axis=1
            ) < 2 * self.minimum_players_per_team
        ]
        self.dataset = self.dataset.drop(corrupted.index, axis=0).reset_index(drop= True)

    #Converts a dataframe of dataset to numpy array
    def DatasetDataframetoNumpy(self, dataframe: pd.DataFrame):
        hteams = np.array(dataframe[self.HOME_TEAM_KEY].tolist())
        ateams = np.array(dataframe[self.AWAY_TEAM_KEY].tolist())
        hplayers = np.array([lineup for lineup in dataframe[self.HOME_LINEUP_KEY].tolist()])
        aplayers = np.array([lineup for lineup in dataframe[self.AWAY_LINEUP_KEY].tolist()])
        return hteams, ateams, hplayers, aplayers

    def _gen_entities(self):
        nodes = self.DatasetDataframetoNumpy(self.dataset)
        self.entities: np.ndarray = np.unique(np.concatenate((
            nodes[0],
            nodes[1],
            np.char.strip(self.nodes[2].reshape(-1), '123456789., '),
            np.char.strip(self.nodes[3].reshape(-1), '123456789., ')
        )))
        #Giving each entity an ID
        self.labeler = skp.LabelEncoder().fit(self.entities)

        teams = np.unique(np.concatenate((
            nodes[0],
            nodes[1]
        )))
        self.team_play_list = {t:[] for t in teams}

        players = np.unique(np.concatenate((
            np.char.strip(self.nodes[2].reshape(-1), '123456789., '),
            np.char.strip(self.nodes[3].reshape(-1), '123456789., ')
        )))
        self.player_play_list = {p:[] for p in players}

