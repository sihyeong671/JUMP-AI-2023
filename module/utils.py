import os
import random
import numpy as np
import torch

def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)