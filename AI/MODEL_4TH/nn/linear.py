import numpy as np
from .module import Module

class LinearModel(Module):
  def __init__(self, a:float=0, b:float=0) -> None:
    self.a = a
    self.b = b
  def forward(self, x:np.array) -> np.array:
    self._x = x
    return self.a*x + self.b
  def backward(self, grad:np.array) -> np.array:
    self._d_a = self._x * grad
    self._d_b = grad
    return self.a * grad
  def update(self, lr:float) -> None:
    self.a -= lr * self._d_a.mean()
    self.b -= lr * self._d_b.mean()