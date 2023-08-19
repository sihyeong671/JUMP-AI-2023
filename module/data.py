import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CustomDataset_v0(Dataset):
  def __init__(self, features: pd.DataFrame, target: np.ndarray, transform=None, is_test=False):
    self.features = features
    self.target = target # HLM or MLM
    self.is_test = is_test # train,valid / test
    
    morgan_fps = np.stack(features["Morgan_FPs"])
    self.train_features = np.append(self.features.drop(columns="Morgan_FPs"), morgan_fps, axis=1)

  def __getitem__(self, index):
    x = self.train_features[index]
    if not self.is_test: # test가 아닌 경우(label 존재)
      label = self.target[index]
      return torch.tensor(x).float(), torch.tensor(label).float() # feature, label
    else: # test인 경우
      return torch.tensor(x).float() # feature
      
  def __len__(self):
    return len(self.features)