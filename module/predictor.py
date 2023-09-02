import os
import sys
from copy import deepcopy


from tqdm import tqdm
import pandas as pd
import numpy as np

import wandb

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from module.utils import Config
from module.data import CustomDataset_v0
from module.model import BaseModel

class Predictor:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
  
  
  def setup(self):
    
    ## submission
    self.submit = pd.read_csv(self.config.SUBMIT_DATAPATH)
    
    ## setup_data
    test_df = pd.read_csv(self.config.TEST_DATAPATH)
    # 결측치 채우기
    test_df["AlogP"] = np.where(pd.isna(test_df["AlogP"]), test_df["LogD"], test_df["AlogP"])
    test_df["mol"] = test_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    test_df.drop(columns=["id", "SMILES"], inplace=True)
    # origin_train_features = train_df[["AlogP", "Molecular_Weight", "Num_H_Acceptors", "Num_H_Donors", "Num_RotatableBonds", "LogD", "Molecular_PolarSurfaceArea"]].values

    fmgen = rdFingerprintGenerator.GetMorganGenerator()
    test_df["Morgan_FPs"] = test_df["mol"].apply(lambda x: fmgen.GetFingerprintAsNumPy(x))
    
    test_features = test_df.drop(columns=["mol"])
    # target = train_df["MLM"].values
    
    # transform
    # train_transform = VarianceThreshold(threshold=0.05)


    test_dataset = CustomDataset_v0(test_features, None, is_test=True)
    
    self.test_dataloader = DataLoader(
      dataset=test_dataset,
      shuffle=False, 
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )
  
    
    ## setup model & loss_fn & optimizer & lr_scheduler
    # self.model = BaseModel(input_size=2055, hidden_size=1024, dropout_rate=0.2, out_size=1)
    self.model = torch.load(f"ckpt/{self.config.MODEL_NAME}_{self.config.DETAIL}.pth")
    
  def predict(self):
    self.model.eval()
    results = []
    with torch.no_grad():
      for inputs in tqdm(self.test_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        
        outputs = self.model(inputs).squeeze(dim=-1)
        results += outputs.cpu().numpy().tolist()
      
    self.submit["HLM"] = results
    
    self.submit.to_csv("csv/MLP_submission.csv", index=False)

        