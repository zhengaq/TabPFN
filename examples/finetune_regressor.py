"""An example for finetuning TabPFN on the California Housing Regression dataset."""
import json
from functools import partial
from pathlib import Path

import sklearn
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn.regressor import TabPFNRegressor
from tabpfn.utils import collate_for_tabpfn_dataset


def eval_test_regression(reg, my_dl_test, lossfn, device):
    """Evaluates the regressor on the test dataloader."""
    reg.model_.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_mse = 0.0
    total_items = 0
    with torch.no_grad(): # No gradients needed for evaluation
        for data_batch in my_dl_test:
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            reg.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = reg.predict_from_preprocessed(X_tests) # Shape: [Batch, NumTestSamples]
            y_tests_dev = y_tests.to(device).float()
            print(f"Shape preds: {preds.shape}")
            print(f"Shape y_tests_dev: {y_tests_dev.shape}")
            #loss = lossfn(preds, y_tests_dev)
            loss = lossfn(preds.view(-1), y_tests_dev.view(-1))
            total_loss += loss.item() * y_tests.numel() # Accumulate total loss
            mse = mean_squared_error(y_tests_dev.cpu().numpy().flatten(),
                                     preds.cpu().numpy().flatten())
            total_mse += mse * y_tests.numel() # Accumulate sum of squared errors * count
            total_items += y_tests.numel()

    avg_loss = total_loss / total_items if total_items > 0 else 0
    avg_mse = total_mse / total_items if total_items > 0 else 0
    reg.model_.train() # Set model back to training mode
    return avg_loss, avg_mse


if __name__ == "__main__":
    device = "cpu"
    n_use =500
    do_epochs = 1

    
    
    # Optional: Scale target for potentially better performance/stability,
    # although TabPFN Regressor handles standardization internally.
    # y_scaler = StandardScaler()
    # y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    data_frame_x, data_frame_y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    splitfn = partial(train_test_split, test_size=0.3)
    X_train, X_test, y_train, y_test = splitfn(data_frame_x[:n_use], data_frame_y[:n_use])
    print(f"Train shape: X-{X_train.shape}, Test shape: X-{X_test.shape}")


    clf = TabPFNRegressor(ignore_pretraining_limits=True, device=device, n_estimators=2,
                           random_state=2, inference_precision=torch.float32)

    datasets_list = clf.get_preprocessed_datasets(X_train, y_train, splitfn, max_data_size=1000)
    datasets_list_test = clf.get_preprocessed_datasets(X_test, y_test, splitfn, max_data_size=1000)
    my_dl_train = DataLoader(datasets_list, batch_size=2, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(datasets_list_test, batch_size=1, collate_fn=collate_for_tabpfn_dataset)

    optim_impl = Adam(clf.model_.parameters(), lr=1e-5)
    lossfn = torch.nn.MSELoss()
    loss_batches = []
    mse_batches = []

    loss_test, res_mse = eval_test_regression(clf, my_dl_test, lossfn, device)
    print("Initial MSE:", res_mse)
    for epoch in range(do_epochs):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            # X_trains, X_tests, y_trains, y_test, cat_ixs, conf, renormalized_criterion, 
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_from_preprocessed(X_tests)
            print(f"Shape preds: {preds.view(-1).shape}")
            print(f"Shape y_tests_dev: {y_tests.view(-1).shape}")
            loss = lossfn(preds.view(-1), y_tests.view(-1).to(device))
            loss.backward()
            optim_impl.step()

        loss_test, res_mse = eval_test_regression(clf, my_dl_test, lossfn, device)
        print(f"---- EPOCH {epoch}: ----")
        print("Test MSE:", res_mse)
        print("Test Loss:", loss_test)

        loss_batches.append(loss_test)
        mse_batches.append(res_mse)
        with Path("finetune_regression.json").open(mode="w") as file:
            json.dump({"loss": loss_batches, "acc": mse_batches}, file)

