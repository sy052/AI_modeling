import numpy as np

class Module:
  def forward(self, x:np.array) -> np.array:
    raise NotImplementedError
  def backward(self, grad:np.array) -> np.array:
    raise NotImplementedError
  def __call__(self, x:np.array) -> np.array:
    return self.forward(x)