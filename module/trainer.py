import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.utils import Config
from module.data import CustomDataset_v0

class Trainer:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
  
  
  def setup_data(self):
    train_df = pd.read_csv(self.config.TRAIN_DATAPATH)
    train_df["mol"] = train_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    train_df.drop(columns=["id", "SMILES"], inplace=True)
    # origin_train_features = train_df[["AlogP", "Molecular_Weight", "Num_H_Acceptors", "Num_H_Donors", "Num_RotatableBonds", "LogD", "Molecular_PolarSurfaceArea"]].values

    fmgen = rdFingerprintGenerator.GetMorganGenerator()
    train_df["Morgan_FPs"] = train_df["mol"].apply(lambda x: fmgen.GetFingerprintAsNumPy(x))
    
    train_features = train_df.drop(columns=["HLM", "MLM", "mol"])
    target = train_df["HLM"].values
    # target = train_df["MLM"].values
    
    # transform
    # train_transform = VarianceThreshold(threshold=0.05)
    
    # split
    train_x, val_x, train_y, val_y = train_test_split(train_features, target, train_size=0.2, random_state=self.config.SEED)

    train_dataset = CustomDataset_v0(train_x, train_y)
    val_dataset = CustomDataset_v0(val_x, val_y)
    
    self.train_dataloader = DataLoader(
      dataset=train_dataset,
      shuffle=True, 
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )

  def train(self):
    pass
  
  
  
  