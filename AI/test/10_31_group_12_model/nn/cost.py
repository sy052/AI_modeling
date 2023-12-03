from nn.model import *
from nn.linear import *

# 평균제곱오차 함수
def mse(y_hat:np.array, y_true:np.array) -> float:
  '''summary

  Args:
      y_hat (np.array): 추정치
      y_true (np.array): 실제값

  Returns:
      E [ (y^ - y)^2 ]
      float으로 리턴 
  '''
  assert len(y_hat) == len(y_true)
  return ((y_hat - y_true)**2).mean()

def grad_mse(model:Module, x:np.array, y_true:np.array) -> dict[str,float]:
  """
  평균제곱오차를 편미분해서 값을 리턴해주는 함수
  
  summary

  Args:
      model (Module): 상속받은 model의 default값 
      x (np.array): 추정치
      y_true (np.array): 실제값

  Returns:
      dict[str,float]: 
      d_w = w의 편미분
      d_b = b의 편미분
      d_w, d_b 리턴

  """
  assert len(x) == len(y_true)
  n = len(x)
  y_hat = model(x)
  d_w = 2*(x*(y_hat-y_true)).mean()
  d_b = 2*(y_hat-y_true).mean()
  return {'d_w': d_w, 'd_b':d_b}