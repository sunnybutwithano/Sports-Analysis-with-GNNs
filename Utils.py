import enum
import torch


class GLOBALS(enum.Enum):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SavePath = 'SavePath/'
    LoadPath = 'LoadPath/'
    SaveEvery = 10


class HYPERPARAETERS(enum.Enum):
    LearningRate = 0.001
    EmbeddingDim = 8
    ConvDims = [8, 16, 16, 16]
    FCDims = [32, 16, 8,  3]
    DropOuts = {'emb': 0.5, 'conv': 0.4, 'fc': 0.4}