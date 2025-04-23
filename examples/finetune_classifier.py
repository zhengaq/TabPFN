"""An example for finetuning TabPFN on the Covertype dataset."""
import json
from functools import partial
from pathlib import Path

import sklearn
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy

from tabpfn import TabPFNClassifier
from tabpfn.utils import collate_for_tabpfn_dataset
from tabpfn.base import ClassifierModelSpecs

import time


def eval_test_old(clf, my_dl_test, lossfn):
    with torch.no_grad():
        loss_sum = 0.0
        acc_sum = 0.0
        acc_items = 0
        for data_batch in my_dl_test:
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss_sum += lossfn(torch.log(preds), y_tests.to(device)).item()
            acc_sum += accuracy_score(y_tests.flatten().cpu(), preds[:,1,:].flatten().cpu()>0.5)*y_tests.numel()
            acc_items += y_tests.numel()

        res_accuracy = acc_sum/acc_items
        return loss_sum, res_accuracy
    
def eval_test(clf: TabPFNClassifier, 
            X_train_raw: np.ndarray, y_train_raw: np.ndarray,
            X_test_raw: np.ndarray, y_test_raw: np.ndarray
        ):
    if hasattr(clf, 'model_') and clf.model_ is not None:
        new_model = copy.deepcopy(clf.model_) # <--- Need!!!
        new_config = copy.deepcopy(clf.config_) #probs do not need
        model_spec_obj = ClassifierModelSpecs(
            model=new_model,
            config=new_config,
            #norm_criterion=None,
        )
        clf_eval = TabPFNClassifier(model_path=model_spec_obj)
    else: 
        clf_eval = TabPFNClassifier()

    clf_eval.fit(X_train_raw, y_train_raw)
    
    try:
        predictions_labels = clf_eval.predict(X_test_raw)
        predictions_proba = clf_eval.predict_proba(X_test_raw) # Needed for log_loss

        accuracy = accuracy_score(y_test_raw, predictions_labels)
        nll = log_loss(y_test_raw, predictions_proba)

    except Exception as e:
        print(f"Error during evaluation prediction/metric calculation: {e}")
        accuracy, nll= np.nan, np.nan
        

    return accuracy, nll

if __name__ == "__main__":
    device = "cpu"
    #n_use = 200_000
    n_use =1000
    do_epochs = 3
    random_seed = 42 
    test_set_size=0.3

    # Load Covertype Dataset (7-way classification)
    data_frame_x, data_frame_y = sklearn.datasets.fetch_covtype(return_X_y=True, shuffle=True)
    splitfn = partial(train_test_split, test_size=0.3)
    indices = np.arange(len(data_frame_y))
    np.random.seed(random_seed) # for reproducibility of subset selection
    subset_indices = np.random.choice(indices, size=n_use, replace=False)
    X_subset = data_frame_x[subset_indices]
    y_subset = data_frame_y[subset_indices]

    X_train, X_test, y_train, y_test = splitfn(data_frame_x[:n_use], data_frame_y[:n_use],
                                               test_size=test_set_size,
                                            stratify=y_subset,
                                            random_state=random_seed,
                                               )

    clf = TabPFNClassifier(ignore_pretraining_limits=True, device=device, n_estimators=2,
                           random_state=2, inference_precision=torch.float32)

    datasets_list = clf.get_preprocessed_datasets(X_train, y_train, splitfn, 1000)
    datasets_list_test = clf.get_preprocessed_datasets(X_test, y_test, splitfn, 1000)
    my_dl_train = DataLoader(datasets_list, batch_size=2, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(datasets_list_test, batch_size=2, collate_fn=collate_for_tabpfn_dataset)

    optim_impl = Adam(clf.model_.parameters(), lr=1e-5)
    lossfn = torch.nn.NLLLoss()
    loss_batches = []
    acc_batches = []

    #loss_test, res_acc = eval_test(clf, my_dl_test, lossfn)
    res_acc, loss_test = eval_test(clf, X_train, y_train, X_test, y_test)
    print("Initial accuracy:", res_acc)
    print("Initial Test Loss:", loss_test)
    for epoch in range(do_epochs):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss = lossfn(torch.log(preds), y_tests.to(device))
            loss.backward()
            optim_impl.step()

        #loss_test, res_acc = eval_test(clf, my_dl_test, lossfn)
        res_acc, loss_test = eval_test(clf, X_train, y_train, X_test, y_test)
        print(f"---- EPOCH {epoch}: ----")
        print("Test Acc:", res_acc)
        print("Test Loss:", loss_test)

        loss_batches.append(loss_test)
        acc_batches.append(res_acc)
        with Path("finetune.json").open(mode="w") as file:
            json.dump({"loss": loss_batches, "acc": acc_batches}, file)

