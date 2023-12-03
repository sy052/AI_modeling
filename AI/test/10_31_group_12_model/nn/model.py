import numpy as np

class Module:   # 기본 학습 모델 모듈
  """summary여러 모델들의 default class
  
    상속받을 class(자식)는 각 메서드를 정의해야하며
    forward 메서드는 call함수로 접근이 됩니다.
  """
  def __init__(self) -> None:   # 클래스 호출 시, 처음 시행되는 함수
    raise NotImplementedError

  def set_params(self) -> None:
    raise NotImplementedError
  
  def get_params(self) -> dict:
    raise NotImplementedError
  
  def forward(self, x:np.array) -> np.array:
    raise NotImplementedError
  
  def __call__(self, x:np.array) -> np.array:   # 클래스 호출 시, forward함수 호출
    return self.forward(x)

