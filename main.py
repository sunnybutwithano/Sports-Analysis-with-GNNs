import pandas as pd
import json
from Utils import gen_entities, Globals
from GNN import HeteroGNN
from torch.optim import Adam
from torch.nn import NLLLoss
from Learning import continuousGrowingTrainer, continuousWindowTrainer, continuousWindowTester


dataset_filename = 'data/FakeData_EPL.csv'
# dataset_filename = 'data/KaggleDataset.csv'

dataset = pd.read_csv(
    dataset_filename,
    encoding='latin-1',
    usecols=['league', 'season', 'week', 'home_team', 'away_team', 'result', 'home_lineup', 'away_lineup']
)
corrupted = dataset.loc[pd.isna(dataset['away_lineup']) | pd.isna(dataset['home_lineup'])]
dataset = dataset.drop(corrupted.index, axis=0).reset_index(drop=True)


with open('hyperparameters.json', 'r') as hp_file:
    hyperparameters = json.load(hp_file)

learning_rate = hyperparameters["learning_rate"]
num_epochs = hyperparameters["num_epochs"]
fc_dropout = hyperparameters["fc_dropout"]
conv_dropout = hyperparameters["conv_dropout"]
emb_dropout = hyperparameters["emb_dropout"]


entities = gen_entities(dataset)

model = HeteroGNN(
    embedding_dims=(
        max(entities.values()) + 1,
        hyperparameters["embedding_dim"]
    ),
    conv_dims=hyperparameters["conv_dims"],
    fully_connected_dims=hyperparameters["fully_connected_dims"],
    dropout={
        "emb": emb_dropout,
        "conv": conv_dropout,
        "fc": fc_dropout
    }
).to(Globals.DEVICE.value)

print(model)

optimizer = Adam(
    model.parameters(),
    lr=learning_rate
)
criterion = NLLLoss()

for league, leaguedf in dataset.groupby('league'):
    print('Training On:', league)
    train_data = leaguedf.iloc[: int(leaguedf.shape[0]* 0.9) - 1, :]
    test_data = leaguedf.iloc[int(leaguedf.shape[0]* 0.9):, :]
    continuousGrowingTrainer(leaguedf, entities, model, optimizer, criterion, num_epochs, 10)
    # continuousWindowTrainer(train_data, entities, model, optimizer, criterion, num_epochs, 30, 10)

    print('Testing On:', league)
    continuousWindowTester(leaguedf, test_data, entities, model, optimizer, criterion, 0, 30, 30)

