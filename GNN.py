import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data

import rdkit
import rdkit.Chem as Chem

from collections import defaultdict


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def create_encoders(df):
  encoder_atom = defaultdict(lambda : len(encoder_atom))
  encoder_bond_type = defaultdict(lambda : len(encoder_bond_type))
  encoder_bond_stero = defaultdict(lambda : len(encoder_bond_stero))
  encoder_bond_type_stero = defaultdict(lambda : len(encoder_bond_type_stero))
  
  target = df["SMILES"].values

  for smiles in tqdm(target):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    
    for atom in m.GetAtoms():
      encoder_atom[atom.GetAtomicNum()]
      
    for bond in m.GetBonds():
      encoder_bond_type[bond.GetBondTypeAsDouble()]
      encoder_bond_stero[bond.GetStereo()]
      encoder_bond_type_stero[(bond.GetBondTypeAsDouble(), bond.GetStereo())]
      
  return encoder_atom, encoder_bond_type, encoder_bond_stero, encoder_bond_type_stero

encoder_atom, encoder_bond_type, encoder_bond_stero, encoder_bond_type_stero = create_encoders(train)

def row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stero, encoder_bond_type_stero):
  smiles = row.SMILES
  MLM = row.MLM
  
  m = Chem.MolFromSmiles(smiles)
  m = Chem.AddHs(m)
  
  num_nodes = len(list(m.GetAtoms()))
  x = np.zeros((num_nodes, len(encoder_atom.keys())))
  for i in m.GetAtoms():
    x[i.GetIdx(), encoder_atom[i.GetAtomicNum()]] = 1
  
  x = torch.from_numpy(x).float()
  
  i = 0
  num_edges = 2 * len(list(m.GetBonds()))
  edge_index = np.zeros((2, num_edges), dtype=np.int64)
  edge_type = np.zeros((num_edges, ), dtype=np.int64)

  for edge in m.GetBonds():
    u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
    v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
    bond_type = edge.GetBondTypeAsDouble()
    bond_stereo = edge.GetStereo()
    bond_label = encoder_bond_type_stero[(bond_type, bond_stereo)]
    
    edge_index[0, i] = u
    edge_index[1, i] = v
    edge_index[0, i+1] = v
    edge_index[1, i+1] = u
    edge_type[i] = bond_label
    edge_type[i+1] = bond_label
    i += 2
    
  
  edge_index = torch.from_numpy(edge_index)
  edge_type = torch.from_numpy(edge_type)

  MLM = torch.tensor([MLM]).float()
  
  data = Data(
    x=x,
    edge_index=edge_index,
    edge_type=edge_type,
    y=MLM,
    uid=row.id
  )
  
  return data

    
  
  
  

