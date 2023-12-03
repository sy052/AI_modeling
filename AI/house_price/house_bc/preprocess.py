from typing import Literal
from dataclasses import dataclass

import pandas as pd

@dataclass
class HomeData:
  file_trn: str = './data/train.csv'
  file_tst: str = './data/test.csv'
  index_col: str = 'Id'
  target_col: str = 'SalePrice'
  drop_cols: tuple[str] = ('LotFrontage', 'MasVnrArea', 'GarageYrBlt')
  fill_num_strategy: Literal['mean', 'min', 'max'] = 'min'

  def _read_df(self, split:Literal['train', 'test']='train'): # train test 선택하고 안하면 train
    if split == 'train':
      df = pd.read_csv(self.file_trn, index_col=self.index_col)
      df.dropna(axis=0, subset=[self.target_col], inplace=True)
      target = df[self.target_col]
      df.drop([self.target_col], axis=1, inplace=True)
      return df, target
    elif split == 'test':
      df = pd.read_csv(self.file_tst, index_col=self.index_col)
      return df
    raise ValueError(f'"{split}" is not acceptable.')

  def preprocess(self):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    trn_df, target = self._read_df('train')
    tst_df = self._read_df('test')

    # drop `drop_cols`
    trn_df.drop(self.drop_cols, axis=1, inplace=True)
    tst_df.drop(self.drop_cols, axis=1, inplace=True)

    # Numerical
    trn_num = trn_df.select_dtypes(exclude=['object'])
    tst_num = tst_df.select_dtypes(exclude=['object'])
    # Categorical
    trn_cat = trn_df.select_dtypes(include=['object'])
    tst_cat = tst_df.select_dtypes(include=['object'])

    # fill the numerical columns using `fill_num_strategy`
    if self.fill_num_strategy == 'mean':
      fill_values = trn_num.mean(axis=1)
    elif self.fill_num_strategy == 'min':
      fill_values = trn_num.min(axis=1)
    elif self.fill_num_strategy == 'max':
      fill_values = trn_num.max(axis=1)
    trn_num.fillna(fill_values, inplace=True)
    tst_num.fillna(fill_values, inplace=True)

    # One-Hot encoding
    enc = OneHotEncoder(dtype=np.float32, sparse_output=False, drop='if_binary', handle_unknown='ignore')
    trn_cat_onehot = enc.fit_transform(trn_cat)
    tst_cat_onehot = enc.transform(tst_cat)

    trn_arr = np.concatenate([trn_num.to_numpy(), trn_cat_onehot], axis=1)
    tst_arr = np.concatenate([tst_num.to_numpy(), tst_cat_onehot], axis=1)

    trn_X = pd.DataFrame(trn_arr, index=trn_df.index)
    tst_X = pd.DataFrame(tst_arr, index=tst_df.index)

    return trn_X, target, tst_X


def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="Data preprocessing", add_help=add_help)
  # inputs
  parser.add_argument("--train-csv", default="./data/train.csv", type=str, help="train data csv file")
  parser.add_argument("--test-csv", default="./data/test.csv", type=str, help="test data csv file")
  # outputs
  parser.add_argument("--output-train-feas-csv", default="./trn_X.csv", type=str, help="output train features")
  parser.add_argument("--output-test-feas-csv", default="./tst_X.csv", type=str, help="output test features")
  parser.add_argument("--output-train-target-csv", default="./trn_y.csv", type=str, help="output train targets")
  # options
  parser.add_argument("--index-col", default="Id", type=str, help="index column")
  parser.add_argument("--target-col", default="SalePrice", type=str, help="target column")
  parser.add_argument("--drop-cols", default=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], type=list, help="drop columns")
  parser.add_argument("--fill-num-strategy", default="min", type=str, help="numeric column filling strategy (mean, min, max)")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  home_data = HomeData(
    args.train_csv,
    args.test_csv,
    args.index_col,
    args.target_col,
    args.drop_cols,
    args.fill_num_strategy
  )
  trn_X, trn_y, tst_X = home_data.preprocess()

  trn_X.to_csv(args.output_train_feas_csv)
  tst_X.to_csv(args.output_test_feas_csv)
  trn_y.to_csv(args.output_train_target_csv)