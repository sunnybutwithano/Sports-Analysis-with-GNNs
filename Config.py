import enum
import torch


class GLOBALS(enum.Enum):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SavePath = 'SavePath/'
    LoadPath = 'LoadPath/'
    SaveEvery = 10

    #Put True to load the Graphs from Load Folder - Put False to make the graphs, save them in Save Folder and use them
    already_saved = True

    #Put True to Load Model and loss Lists from Load Folder - Put False to start a new training
    # ATTENTION: The result of the training will be saved in the Save Folder ANYWAYS!!! - Copy Your Work Before Starting
    continue_training = False


class HYPERPARAETERS(enum.Enum):
    LearningRate = 0.001
    Phase1Rounds = 2
    Phase1Epochs = 50
    Phase2Rounds = 2
    Phase2Epochs = 20
    Phase3Epochs = 2
    EmbeddingDim = 2
    ConvDims = [2, 4, 12]
    FCDims = [24, 8,  3]
    BladeChestDim = 8
    DropOuts = {'emb': 0.5, 'conv': 0.3, 'fc': 0.4}
    ValidationPortion = 0.1
    TestPortion = 0.1