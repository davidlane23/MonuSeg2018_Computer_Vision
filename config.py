import numpy as np
import torch

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCH = 10
BATCH_SIZE = 16
LRATES = [0.01, 0.001]
N_CLASSES = 2
SAVE_DIR = "MonuSeg_results"

np.random.seed(0)
torch.manual_seed(0)
