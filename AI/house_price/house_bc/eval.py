import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from dataclasses import dataclass, field
from typing import Type, Optional
import pandas as pd

def evaluate(
  model:nn.Module,
  data_loader:DataLoader,
  metric:torchmetrics.metric.Metric,
  device:str='cpu',
) -> None:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      metric.update(output, y)

@dataclass
class KFoldCV:
  X: torch.Tensor
  y: torch.Tensor
  Model: Type[nn.Module]
  model_args: tuple = tuple()
  model_kwargs: dict = field(default_factory=lambda : {})
  epochs: int = 500
  criterion: callable = F.mse_loss
  Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
  optim_kwargs: dict = field(default_factory=lambda : {})
  trn_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  val_dl_kwargs: dict = field(default_factory=lambda : {'batch_size': 36})
  n_splits: int = 5
  metric: torchmetrics.Metric = torchmetrics.MeanSquaredError(squared=False)
  device: str = 'cpu'

  def run(self):
    from torch.utils.data import TensorDataset
    from sklearn.model_selection import KFold
    from tqdm.auto import trange
    from train import train_one_epoch

    model = self.Model(*self.model_args, **self.model_kwargs).to(self.device)
    models = [self.Model(*self.model_args, **self.model_kwargs).to(self.device) for _ in range(self.n_splits)]
    for m in models:
      m.load_state_dict(model.state_dict())
    kfold = KFold(n_splits=self.n_splits, shuffle=False)

    metrics = {'trn_rmse': [], 'val_rmse': []}
    for i, (trn_idx, val_idx) in enumerate(kfold.split(self.X)):
      X_trn, y_trn = self.X[trn_idx], self.y[trn_idx]
      X_val, y_val = self.X[val_idx], self.y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, **self.trn_dl_kwargs)
      dl_val = DataLoader(ds_val, **self.val_dl_kwargs)

      m = models[i]
      optim = self.Optimizer(m.parameters(), **self.optim_kwargs)

      pbar = trange(self.epochs)
      for _ in pbar:
        train_one_epoch(m, self.criterion, optim, dl_trn, self.metric, self.device)
        trn_rmse = self.metric.compute().item()
        self.metric.reset()
        evaluate(m, dl_val, self.metric, self.device)
        val_rmse = self.metric.compute().item()
        self.metric.reset()
        pbar.set_postfix(trn_rmse=trn_rmse, val_loss=val_rmse)
      metrics['trn_rmse'].append(trn_rmse)
      metrics['val_rmse'].append(val_rmse)
    return pd.DataFrame(metrics)

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  import numpy as np
  from nn import ANN

  args = get_args_parser().parse_args()
  
  exec(open(args.config).read())
  cfg = config

  train_params = cfg.get('train_params')
  device = train_params.get('device')

  files = cfg.get('files')
  X_df = pd.read_csv(files.get('X_csv'), index_col='Id')
  y_df = pd.read_csv(files.get('y_csv'), index_col='Id')

  X, y = torch.tensor(X_df.to_numpy(dtype=np.float32)), torch.tensor(y_df.to_numpy(dtype=np.float32))

  Model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X.shape[-1]
  
  
  dl_params = train_params.get('data_loader_params')

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')

  metric = train_params.get('metric').to(device)
  
  cv = KFoldCV(X, y, Model, model_kwargs=model_params,
               epochs=train_params.get('epochs'),
               criterion=train_params.get('loss'),
               Optimizer=Optim,
               optim_kwargs=optim_params,
               trn_dl_kwargs=dl_params, val_dl_kwargs=dl_params,
               metric=metric,
               device=device)
  res = cv.run()

  res = pd.concat([res, res.apply(['mean', 'std'])])
  print(res)
  res.to_csv(files.get('output_csv'))