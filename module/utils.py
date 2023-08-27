import os
import random
import numpy as np
import torch

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  

class Config:
  def __init__(self, **kwargs):
    self.SEED = kwargs["seed"]
    self.EPOCHS = kwargs["epochs"]
    self.LR = kwargs["lr"]
    self.MODE = kwargs["mode"]
    
    self.NUM_WORKERS = kwargs["num_workers"]
    self.BATCH_SIZE = kwargs["batch_size"]
    self.MODEL_NAME = kwargs["model_name"]
    self.DETAIL = kwargs["detail"]
    self.TRAIN_DATAPATH = kwargs["train_datapath"]
    self.TEST_DATAPATH = kwargs["test_datapath"]
    self.SUBMIT_DATAPATH = kwargs["submit_datapath"]
    
    self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")