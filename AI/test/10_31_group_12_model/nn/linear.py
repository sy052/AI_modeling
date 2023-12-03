from nn import model
import numpy as np

# model클래스를 부모 클래스로 하는 선형회귀 함수 모델
class LinearModel(model.Module):
  """summary 

  Args:
    w:float, b:float을 인자로 받아서 
    set_params(self, w:float, b:float)에 변수로 저장된 후
    get_param에서 {'w': w, 'b':b}로 리턴하며
    forward(self, x:np.array)에서 w, b의 value에 접근되어 
    w * x + b로 리턴됩니다.

  """
  
  # 클래스가 호출되면 먼저 실행되는 함수
  def __init__(self, w:float=.0, b:float=.0) -> None:
    self.set_params(w, b)

  # 입력받은 값를 w,b변수에 입력하는 함수
  def set_params(self, w:float, b:float) -> None:
    self.w = w
    self.b = b

  # 입력받은 w,b 값을 dict자료구조로 저장하는 함수
  def get_params(self) -> dict[str,float]:
    return {'w': self.w, 'b':self.b}
  
  # 부모 클래스의 forward함수에 기능을 추가한 함수
  def forward(self, x:np.array) -> np.array:
    params = self.get_params()
    w = params.get('w')
    b = params.get('b')
    return w * x + b
  
  
  # 가중치를 업데이트하는 함수
def update(model:Module, lr:float, d_w:float, d_b:float) -> None:
  params_old = model.get_params()
  params_new = {
    'w': params_old.get('w') - lr*d_w,
    'b': params_old.get('b') - lr*d_b,
  }
  model.set_params(**params_new)