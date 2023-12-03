import pandas as pd
import numpy as np

#def get_X(df:pd.DataFrame, features:iter=["Age","Sibsp","Parch","Fare","Pclass","Gender","Embarked"]):
def get_X(df:pd.DataFrame, features:iter=['LotArea', 'MSSubClass', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'MoSold', 'YrSold']): 
  '''
  age_mean = train_df["Age"].mean() # 평균
  fare_median = train_df["Fare"].median() # 중앙값
  cabin_unk = "UNK" # 새로운 범주
  embarked_mode = train_df["Embarked"].mode()[0] # 최빈값
  age_mean , fare_median ,cabin_unk , embarked_mode

  train_df["Age"] = train_df["Age"].fillna(age_mean)
  train_df["Cabin"] = train_df["Cabin"].fillna(cabin_unk)

  test_df["Age"] = test_df["Age"].fillna(age_mean)
  test_df["Fare"] = test_df["Fare"].fillna(fare_median)
  test_df["Cabin"] = test_df["Cabin"].fillna(cabin_unk)
  test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)
  '''

    # from https://www.kaggle.com/code/alexisbcook/titanic-tutorial
  return pd.get_dummies(df[features]).to_numpy(dtype=np.float32)

#def get_y(df:pd.DataFrame):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
#  return df.SalePrice.to_numpy(dtype=np.float32)
  
def get_y(df:pd.DataFrame, feature = "SalePrice"):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  return (df[feature] / 10000).to_numpy(dtype=np.float32)