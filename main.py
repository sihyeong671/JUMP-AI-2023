import argparse

from module.utils import Config, seed_everything
from module.trainer import Trainer
from module.predictor import Predictor

def run():
  pass


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--seed', type=int, default=777)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--mode', type=str, default="train")
  parser.add_argument('--num_workers', type=int, default=4) 
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--model_name', type=str, default="MLP")
  parser.add_argument('--train_datapath', type=str, default="data/train.csv")
  parser.add_argument('--test_datapath', type=str, default="data/test.csv")
  parser.add_argument('--submit_datapath', type=str, default="data/sample_submission.csv")
  parser.add_argument('--detail', default="v0")
  args = parser.parse_args()
  
  CONFIG = Config(**vars(args))
  
  seed_everything(CONFIG.SEED)
  
  
  if CONFIG.MODE == "train":
    trainer = Trainer(CONFIG)
    trainer.setup()
    trainer.train()
  
  elif CONFIG.MODE == "predict":
    predictor = Predictor(CONFIG)
    predictor.setup()
    predictor.predict()
