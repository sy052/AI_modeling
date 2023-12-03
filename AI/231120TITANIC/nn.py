import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from utils import CustomDataset

class ANN(nn.Module):
  def __init__(self, input_dim=5, hidden_dim=128):
    super().__init__()
    self.lin1 = nn.Linear(input_dim,hidden_dim)
    self.lin2 = nn.Linear(hidden_dim,1)
    self.dropout = nn.Dropout(0.3)
   
  def forward(self, x):
    #순전파/ sigmoid는 마지막에 사용. 
    # sigmoid 사용하는 이유는 타이타닉이 0/1(이진분류로 보통 sigmoid 함수 사용함)
    x = self.lin1(x)
    x = nn.functional.sigmoid(x)
    x = self.dropout(x)
    x = self.lin2(x)
    x = nn.functional.sigmoid(x)
    return x
  
  # cross validate에 값을 쉽게 넣을 수 있게끔
class ANN_Estimator(ANN, BaseEstimator):
  def __init__(self, hidden=128, optim=torch.optim.Adam, lr=0.0001, loss_fn=nn.functional.binary_cross_entropy, device='cpu'):
    self.hidden = hidden
    self.optim = optim
    self.lr = lr
    self.loss_fn = loss_fn

    super().__init__(self.hidden)
    self.optimizer = self.optim(self.parameters(), lr=self.lr)
    self.device = device
    self.to(device)

#학습
  def fit(self, X, y):
    self.train()

    ds = CustomDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    pbar = tqdm(range(300))
    for _ in pbar:
      for _X, _y in dl:
        _X, _y = _X.to(self.device), _y.to(self.device)
        _pred = self.forward(_X)
        loss = self.loss_fn(_pred, _y.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      pbar.set_postfix(trn_loss=loss.item())

#예측
  def predict(self, X):
    self.eval()
    with torch.no_grad():
      pred = self.forward(torch.tensor(X, device=self.device))
    return (pred.cpu() > 0.5).float().numpy()