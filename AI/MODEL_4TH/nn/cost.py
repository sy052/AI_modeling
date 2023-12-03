import numpy as np
from .module import Module

class MSE(Module):
  def forward(self, y_hat:np.array, y_true:np.array) -> np.array:
    self._hat = y_hat
    self._true = y_true
    return ((y_hat - y_true)**2).mean()
  def backward(self) -> np.array:
    return 2*(self._hat - self._true)
