import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

########## 수정사항 ###############
# 원하는 feature를 선택할 수 있도록 -> 이미 설계되어 있음
# y가 Survived로 고정되어 있음 -> get_X처럼 이름을 설정하면 자동으로 y를 가져올 수 있도록 설계
# Scaler 설계
##################################


from sklearn.preprocessing import LabelEncoder




def get_X(df:pd.DataFrame, features:iter=["Pclass", "Sex", "SibSp", "Parch", "Embarked"], scaler= 'standard', scale_columns = None):
    standard = StandardScaler()
    maxabs = MaxAbsScaler()
    minmax = MinMaxScaler()
    
###스케일링 하는 이유 정규분포가 모델 학습에 도움을 줌으로.(그래프 대칭, 정규화)
    df1 = pd.get_dummies(df[features])
    if scale_columns :
        if scaler == 'standard':
          df1[scale_columns] = standard.fit_transform(df1[scale_columns])
        elif scaler == 'maxabs':
          df1[scale_columns] = maxabs.fit_transform(df1[scale_columns])
        elif scaler == 'minmax':
            df1[scale_columns] = minmax.fit_transform(df1[scale_columns])

    
    
    return df1.to_numpy(dtype=np.float32)

def get_y(df:pd.DataFrame, feature = "Survived"):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  return (df[feature] / 10000).to_numpy(dtype=np.float32) #값의 saleprice 편차가 각각 너무 커서 학습 시 도움을 주기 위해/10000. 스케일링과 같은 느낌
 #추후 output에 *10000