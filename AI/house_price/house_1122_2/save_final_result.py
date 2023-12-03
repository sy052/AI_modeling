import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
import os
import math


def main(cfg):
    import numpy as np
    import pandas as pd
    from torch.utils.data.dataset import TensorDataset
    from nn import ANN
    from tqdm.auto import trange

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    files = cfg.get("files")
    pd_tst = pd.read_csv(files.get("X_tst_csv"), index_col=0)
    X_tst = torch.tensor(pd_tst.to_numpy(dtype=np.float32))

    # dl_params = train_params.get("data_loader_params")
    ds = TensorDataset(X_tst)
    dl = DataLoader(ds)

    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X_tst.shape[-1]
    model = Model(**model_params).to(device)

    # Optim = train_params.get("optim")
    # optim_params = train_params.get("optim_params")
    # optimizer = Optim(model.parameters(), **optim_params)

    # loss = train_params.get("loss")
    # metric = train_params.get("metric")
    # values = []
    # pbar = trange(train_params.get("epochs"))
    # for _ in pbar:
    #     train_one_epoch(model, loss, optimizer, dl, metric, device)
    #     values.append(metric.compute().item())
    #     metric.reset()
    #     pbar.set_postfix(trn_loss=values[-1])

    model.load_state_dict(torch.load(files.get("output")))
    model.eval()
    result = []

    with torch.inference_mode():
        for X in dl:
            X = X[0].to(device)
            output = (
                float((model(X)).squeeze()) * 10000
                if not math.isnan(float((model(X)).squeeze()))
                else 0
            )
            print(output)
            result.append(output)

    test_id = pd_tst.index.tolist()
    col_name = ["Id", "SalePrice"]
    list_df = pd.DataFrame(zip(test_id, result), columns=col_name)
    model_name = os.path.splitext(os.path.basename(files.get("output")))[0]
    result_file_name = f"Result_{model_name}.csv"
    list_df.to_csv(result_file_name, index=False)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Save Final Result for Submission", add_help=add_help
    )
    parser.add_argument(
        "-c", "--config", default="./config.py", type=str, help="configuration file"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    main(config)
