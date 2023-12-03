import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN

config = {
    "files": {
        "X_csv": "./trn_X.csv",
        "y_csv": "./trn_y.csv",
        "output": "./model.pth", #모델 저장(학습된 레이어들,편향, 가중치...)
        "output_csv": "./results/five_fold.csv",
        "X_tst_csv": "./tst_X.csv",
    },
    "model": ANN,
    "model_params": {
        "input_dim": "auto",  # Always will be determined by the data shape
        "hidden_dim": [128, 128, 64, 32],
        "use_dropout": True,
        "dropout_ratio": 0.3,
        "activation": torch.nn.ReLU(),
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 32,
            "shuffle": True,
        },
        "loss": F.mse_loss,
        "optim": torch.optim.Adam,
        "optim_params": {
            "lr": 0.001,
        },
        "metric": torchmetrics.MeanSquaredError(squared=False),
        "device": "cpu",
        "epochs": 30,
    },
    "cv_params": {
        "n_split": 5,
    },
}
