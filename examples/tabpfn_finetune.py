## Debug TabPFN on Adult to better understand training and inference.
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.utils import collate_for_tabpfn_dataset, pad_tensors
import json
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from example_dataloaders import get_adult_preprocessed_inputs
from torch.optim import Adam

def eval_test(clf, my_dl_test, lossfn):
    with torch.no_grad():
        loss_sum = 0.0
        acc_sum = 0.0
        acc_items = 0
        for i, data_batch in enumerate(my_dl_test):
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            y_test_padded = torch.stack(pad_tensors(y_tests, labels=True))
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss_sum += lossfn(torch.log(preds), y_test_padded.to(device)).item()
            acc_sum += accuracy_score(y_test_padded.flatten().cpu(), preds[:,1,:].flatten().cpu()>0.5)*y_test_padded.numel()
            acc_items += y_test_padded.numel()
        
        res_accuracy = acc_sum/acc_items
        print("Test Acc:", res_accuracy)
        print("Test Loss:", loss_sum)
        
        return loss_sum, res_accuracy
    
if __name__ == "__main__":
    n_samples = 200
    device = "cuda:1"
    res, data_adult_train_labels, res_test, data_adult_test_labels, cat_indicses = get_adult_preprocessed_inputs()
    
    clf = TabPFNClassifier(ignore_pretraining_limits=True, device=device, n_estimators=2, 
                           random_state=2, inference_precision=torch.float32)
    
    X_data = [res]
    y_data = [data_adult_train_labels]
    
    splitfn = partial(train_test_split, test_size=0.3)
    
    datasets_list = clf.get_preprocessed_datasets(X_data, y_data, splitfn, True, 1000)
    datasets_list_test = clf.get_preprocessed_datasets([res_test], [data_adult_test_labels], splitfn, True, 1000)
    my_dl_train = DataLoader(datasets_list, batch_size=2, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(datasets_list_test, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    
    optim_impl = Adam(clf.model_.parameters(), lr=1e-5)
    lossfn = torch.nn.NLLLoss()
    loss_batches = []
    acc_batches = []
    steps = 0
    for epochs in range(10):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            y_test_padded = torch.stack(pad_tensors(y_tests, labels=True))
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            loss = lossfn(torch.log(preds), y_test_padded.to(device))
            loss.backward()
            optim_impl.step()
        steps += 1
        
        loss_test, res_acc = eval_test(clf, my_dl_test, lossfn)
        loss_batches.append(loss_test)
        acc_batches.append(res_acc)
        json.dump({"loss": loss_batches, "acc": acc_batches}, open("finetune.json", "w"))

            
        