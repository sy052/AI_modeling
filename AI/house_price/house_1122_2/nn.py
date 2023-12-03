import torch.nn as nn
import torch.nn.functional as F

# class ANN(nn.Module):
#   def __init__(self, input_dim:int=5, hidden_dim:int=128, dropout:float=0.3, activation=F.sigmoid):
#     super().__init__()
#     self.lin1 = nn.Linear(input_dim,hidden_dim)
#     self.lin2 = nn.Linear(hidden_dim,1)
#     self.dropout = nn.Dropout(dropout)
#     self.activation = activation
#   def forward(self, x):
#     x = self.lin1(x)
#     x = self.activation(x)
#     x = self.dropout(x)
#     x = self.lin2(x)
#     return x


class ANN(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: list = [128, 128, 64, 32],
        activation: nn.Module = nn.ReLU(),
        use_dropout: bool = False,
        dropout_ratio: float = 0.5,
    ):
        """Artificial Neural Network(ANN) with Linear layers
          the model structure is like below:

          Linear
          Dropout
          Activation

          Linear
          Dropout
          ...


        Args:
          input_dim (int): dimension of input
          hidden_dim (list): list of hidden dimension. the length of 'hidden_dim' means the depth of ANN
          activation (torch.nn.Module): activation name. choose one of [sigmoid, relu, tanh, prelu]
          use_dropout (bool): whether use dropout or not
          drop_ratio (float): ratio of dropout
        """
        super().__init__()

        dims = [input_dim] + hidden_dim  # [5, 128, 128, 64, 32]

        self.dropout = nn.Dropout(dropout_ratio)
        self.identity = nn.Identity()
        self.activation = activation
        self.relu = nn.ReLU()

        model = [
            [
                nn.Linear(dims[i], dims[i + 1]),
                self.dropout if use_dropout else self.identity,
                self.activation,
            ]
            for i in range(len(dims) - 1)
        ]

        output_layer = [
            nn.Linear(dims[-1], 1),
            self.relu,
        ]  # Delete sigmoid for regression model

        self.module_list = nn.ModuleList(sum(model, []) + output_layer)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x
