import torch
import torch.nn as nn

class BaseModel(nn.Module):
  def __init__(self, input_size, hidden_size, dropout_rate, out_size):
    super().__init__()
    
    # fc 레이어 3개와 출력 레이어
    self.fc1 = nn.Linear(input_size, hidden_size) 
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc_out = nn.Linear(hidden_size, out_size)
    
    # 정규화
    self.ln1 = nn.LayerNorm(hidden_size)
    self.ln2 = nn.LayerNorm(hidden_size)
    self.ln3 = nn.LayerNorm(hidden_size)        
    
    # 활성화 함수
    self.activation = nn.LeakyReLU()
    
    # Dropout
    self.dropout = nn.Dropout(dropout_rate)
    
  def forward(self, x):
    out = self.fc1(x)
    out = self.ln1(out)
    out = self.activation(out)
    out = self.dropout(out)
    
    out = self.fc2(out)
    out = self.ln2(out)
    out = self.activation(out)
    out = self.dropout(out)
    
    out = self.fc3(out)
    out = self.ln3(out)
    out = self.activation(out)
    out = self.dropout(out)

    out = self.fc_out(out)
    return out