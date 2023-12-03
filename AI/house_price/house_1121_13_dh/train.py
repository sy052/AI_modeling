import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional, List
import numpy as np
import pandas as pd

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
  
from preprocess import get_X, get_y
from nn import ANN
from utils import CustomDataset
from tqdm.auto import tqdm

#학습
def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    # print(output)
    # print(y)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset) #전체 데이터값에 대한 loss값(손실)의 평균(합) 구하기
#MSE와 다르게 위는 배치사이즈까지 고려한 것

#평가
def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
  multi_metrics: List[torchmetrics.metric.Metric]=None
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval() #평가시작
  total_loss = 0.
  with torch.inference_mode(): #평가부분이므로 기울기를 계산하지 않겠다고 선언하는 부분이므로 역전파 과정은X
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y) #output만 구하는 과정
      if metric is not None:
        metric.update(output, y)
      if multi_metrics is not None:
        for metric in multi_metrics:
          metric.update(output, y) 
  return total_loss/len(data_loader.dataset)

#한번은 검증 나머지는 평가 과정을 거치기 위해(kfold) 위 함수를 evaluate함수로 나눔
#kfold는 대부분 평균값
def kfold_cross_validate(model: nn.Module, criterion:callable, device:str, X_trn:np.array, y_trn:np.array, n_splits:int=5):
  from sklearn.model_selection import KFold #straitfied KFold : ㄱ각각 비율을 맞춤 이진분류에 보통 사용하며 그냥 KFold는 비율이 항상 맞진 않기에 이를 더 선호함.
  from copy import deepcopy
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  nets = [deepcopy(model) for i in range(n_splits)]
  scores = {
  #오차를 알아냄. 음수를 모두 양수로 바꾸기 위함.
  'MSE': [], #ex)-1을 1로바꿈. 제곱값의 평균(오차)
  'MAE': [], #절대값의 평균 
  'RMSE': [], #제곱값의 제곱근, 유클리디안 거리:대각선형태, 맨해튼계단 피타고라스의 정리와 같이 빗변의 길이와 같은 느낌
              #L1정규화 :Lasso(절대값을 사용해서 가중치를 0으로), L2정규화:Ridge(가중치를 0에 가깝게)
  }
  
  for i, (trn_idx, val_idx) in enumerate(kf.split(X_trn)): #녹색은 클래스, 노란색은 함수선언
    X, y = torch.tensor(X_trn[trn_idx]), torch.tensor(y_trn[trn_idx]) #train data X,y데이터를 텐서 배열로 만들기 위함
    X_val, y_val = torch.tensor(X_trn[val_idx]), torch.tensor(y_trn[val_idx])
    ds = CustomDataset(X, y) #X, y를 튜플형태로 감싸는 것 = 합침
    ds_val = CustomDataset(X_val, y_val)
    dl = DataLoader(ds, batch_size=32, shuffle=True) #데이터로더 클래스에서 미니배치 32개로 나눔
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    net = nets[i].train() #nets = [deepcopy(model) for i in range(n_splits)] kf split한 값이 들어감. i에는 0,1,2,3,4

    pbar = tqdm(range(args.epochs)) #bar형태로 나타내줌 pbar=300이니 에폭을 300한다
    for j in pbar:
      mae, mse, rmse = MeanAbsoluteError().to(device), MeanSquaredError().to(device), MeanSquaredError(squared=False).to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) #optimizer는 경사하강법으로
      loss = train_one_epoch(net, criterion, optimizer, dl, device) 
      loss_val = evaluate(net, criterion, dl_val, device, multi_metrics = [mae, mse, rmse]) #multi_metrics: 세개 모두 넣기 위해 선언함.
      mae_val, mse_val, rmse_val = mae.compute().item(), mse.compute().item(), rmse.compute().item() #compute().item() : 가져온다는 뜻
      pbar.set_postfix(trn_loss=loss, val_loss=loss_val) # 진행바 우측에 진행상황 표시
    scores["MAE"].append(mae_val) 
    scores["MSE"].append(mse_val)
    scores["RMSE"].append(rmse_val)
    
  return scores


def main(args): #모든 정보 다 들어있음.
  device = torch.device(args.device)

  train_df = pd.read_csv(args.data_train)
  test_df = pd.read_csv(args.data_test)
  
  feature_list = ['OverallQual', 'GrLivArea', 'GarageCars', 'ExterQual',
       'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'KitchenQual',
       'FullBath',  'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Fireplaces',
       'HeatingQC', 'BsmtFinSF1', 'Foundation', 'WoodDeckSF',
       '2ndFlrSF']
  X_trn = get_X(train_df, scaler=args.scaler, features=feature_list) #데이터 전처리 ->  return df1.to_numpy(dtype=np.float32)
  y_trn = get_y(train_df, feature="SalePrice")[:,np.newaxis]
  X_tst = get_X(test_df, scaler=args.scaler, features=feature_list)

  ds = CustomDataset(X_trn, y_trn) # X_trn, y_trn 값을 훈련 데이터에 합침 (data loader 위함)
  dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle) #배치사이즈를 나누기 위해 사용하는 라이브러리(dataloader)

  ds_tst = CustomDataset(X_tst)
  dl_tst = DataLoader(ds_tst, batch_size=args.batch_size)

  model = ANN(X_trn.shape[-1], hidden_dim=args.hidden_dim, activation=args.activation, use_dropout=args.use_dropout, drop_ratio = args.drop_ratio).to(device)
  print(model) #shape[-1] 입력 특성 갯수를 고정하기 위해 -1을 함. 1로 해도 되나, 이미지 트레이닝 시 보통 -1 사용.
  loss_func = nn.MSELoss() #nn 라이브러리 자체를 loss_func 함수로 지정 MSE로
  
  # k-fold cross validation
  scores = kfold_cross_validate(model, loss_func, device, X_trn, y_trn)
  mean_scores = {k:sum(v) / len(v) for k, v in scores.items()} #key, value값을 뽑아서 for 문으로 , items:key,value를 배출하기 위해
  #key는 dictionary 구조.={} 
  print(mean_scores)

  # train with full trainset
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  pbar = range(args.epochs)
  if args.pbar:
    pbar = tqdm(pbar) #tqbm:pbar를 시각적으로 보여주는 라이브러리 터미널에서 하얀 부분
  for _ in pbar:
    loss = train_one_epoch(model, loss_func, optimizer, dl, device)
    pbar.set_postfix(trn_loss=loss) #trn_loss가 터미널에서 내려가면서 학습이 잘 되는 지 확인
  
  # save pretrained weight
  torch.save(model.state_dict(), args.output) #model.state_dict():레이어들의 편향, 가중치가 dict형태로 저장되어 있음.
  #args:아래 argument에서 output 부분으로 값이 "./model.pth"로 저장됨.
  # final outuput with testset
  '''
  model = ANN(X_trn.shape[-1], hidden_dim=args.hidden_dim, activation=args.activation, use_dropout=args.use_dropout, drop_ratio = args.drop_ratio).to(device)
  model.load_state_dict(torch.load(args.output))
  model.eval()
'''
  # result = []
  # with torch.inference_mode():
  #   for X in dl_tst:
  #     X = X[0].to(device)
  #     output = torch.where(model(X).squeeze() > 0.5, 1, 0).tolist()
  #     result.extend(output)
  
  # print(result)

  # test_id = test_df.PassengerId.tolist()
  # col_name = ['PassengerId', 'Survived']
  # list_df = pd.DataFrame(zip(test_id, result), columns=col_name)
  # list_df.to_csv("Result.csv", index=False)


def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

  parser.add_argument("--data-train", default="./data/train.csv", type=str, help="train dataset path")
  parser.add_argument("--data-test", default="./data/test.csv", type=str, help="test dataset path")
  parser.add_argument("--device", default="cpu", type=str, help="device (Use cpu/cuda/mps)")
  parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
  parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
  parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
  parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
  parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
  parser.add_argument("-o", "--output", default="./model.pth", type=str, help="path to save output model")

  ## model setting
  parser.add_argument("--hidden-dim", default=[128, 128, 64, 32], type=int, nargs='+', help="list of dimension of hidden layer")
  parser.add_argument("--activation", default="relu", choices=["sigmoid", "relu", "tanh", "prelu"])
  parser.add_argument("--use_dropout", action='store_true')
  parser.add_argument("--drop_ratio", default=0.5, type=float)

  ## preprocess setting
  parser.add_argument("--scaler", default="standard", choices=["standard", "minmax", "maxabs"])
  # standard scaler: 평균0, 표준편차1, minmax scaler: 최댓값1, 최솟값0 => scaler: 0과 1사이 값으로 만듬
  
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  main(args)