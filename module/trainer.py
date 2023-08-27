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

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from module.utils import Config
from module.data import CustomDataset_v0
from module.model import BaseModel

class Trainer:
  def __init__(self, CONFIG: Config):
    self.config = CONFIG
  
  
  def setup(self):
    
    ## setup_data
    train_df = pd.read_csv(self.config.TRAIN_DATAPATH)
    train_df = train_df.dropna() # null 있는 두개의 행 제거 
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
    train_x, val_x, train_y, val_y = train_test_split(train_features, target, test_size=0.2, random_state=self.config.SEED)

    train_dataset = CustomDataset_v0(train_x, train_y)
    val_dataset = CustomDataset_v0(val_x, val_y)
    
    self.train_dataloader = DataLoader(
      dataset=train_dataset,
      shuffle=True, 
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )
    
    self.val_dataloader = DataLoader(
      dataset=val_dataset,
      shuffle=False,
      batch_size=self.config.BATCH_SIZE,
      num_workers=self.config.NUM_WORKERS
    )
    
    ## setup model & loss_fn & optimizer & lr_scheduler
    self.model = BaseModel(input_size=2055, hidden_size=1024, dropout_rate=0.2, out_size=1)
    
    self.loss_fn = nn.MSELoss()
    
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer=self.optimizer,
      mode="min",
      factor=0.5,
      patience=10,
      min_lr=1e-5
    )
    
    ## setup wandb
    
    wandb.init(
      entity="bsh",
      project="dacon-ai-drug",
      name=f"{self.config.MODEL_NAME}_{self.config.DETAIL}",
      config={
        "architecture": self.config.MODEL_NAME,
        "epochs": self.config.EPOCHS,
      }
    )

  def train(self):
    self.model.to(self.config.DEVICE)
    
    best_model = None
    best_val_loss = sys.maxsize
    
    for epoch in range(1, self.config.EPOCHS+1):
      train_loss = 0
      self.model.train()
      for inputs, labels in tqdm(self.train_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        labels = labels.to(self.config.DEVICE)

        self.optimizer.zero_grad()
        
        outputs = self.model(inputs).squeeze(dim=-1)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        # update clip
        self.optimizer.step()
        
        train_loss += loss.item()
      
      train_loss /= len(self.train_dataloader)

      val_loss = self._valid()
      
      if self.scheduler is not None:
        self.scheduler.step(val_loss)
      
      wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": self.optimizer.param_groups[0]['lr']
      })

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(self.model)
    
    os.makedirs("ckpt/", exist_ok=True)
    torch.save(best_model, f"ckpt/{self.config.MODEL_NAME}_{self.config.DETAIL}.pth")
      
  def _valid(self):
    self.model.eval()
    with torch.no_grad():
      val_loss = 0
      for inputs, labels in tqdm(self.val_dataloader):
        inputs = inputs.to(self.config.DEVICE)
        labels = labels.to(self.config.DEVICE)
  
        outputs = self.model(inputs).squeeze(dim=-1)
        loss = self.loss_fn(outputs, labels)
        
        val_loss += loss.item()
      
      return val_loss / len(self.val_dataloader)
      