import enum
import torch


class GLOBALS(enum.Enum):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SavePath = 'SavePath/'
    LoadPath = 'LoadPath/'
    # LoadPath = 'Runs/26Jan_2/'
    SaveEvery = 10


class HYPERPARAETERS(enum.Enum):
    LearningRate = 0.001
    Phase1Rounds = 2
    Phase1Epochs = 50
    Phase2Rounds = 2
    Phase2Epochs = 20
    Phase3Epochs = 2
    EmbeddingDim = 8
    ConvDims = [8, 16, 16, 16]
    FCDims = [32, 16, 8,  3]
    DropOuts = {'emb': 0.5, 'conv': 0.4, 'fc': 0.4}