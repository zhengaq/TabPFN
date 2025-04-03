## Debug TabPFN on Adult to better understand training and inference.
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

## Feature ordinal encoding for Gradient boosting
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import json

from torch.optim import Adam
from example_dataloaders import get_adult_preprocessed_inputs


if __name__ == "__main__":
    n_samples = 200 # samples used for prompt_tuning
    
    device = "cuda:1"
    res, data_adult_train_labels, res_test, data_adult_test_labels, cat_indicses = get_adult_preprocessed_inputs()
    clf = TabPFNClassifier(ignore_pretraining_limits=True, device=device, n_estimators=1,
                           random_state=2, inference_precision=torch.float32, differentiable_input=True)
    
    input_x_tensor = torch.tensor(res[:n_samples], dtype=torch.float).to(device)
    input_y_tensor = torch.tensor(data_adult_train_labels[:n_samples], dtype=torch.float).to(device)
    
    train_x_tensor = torch.tensor(res[n_samples:], dtype=torch.float).to(device)
    train_y_tensor = torch.tensor(data_adult_train_labels[n_samples:].values, dtype=torch.long).to(device)
    
    query_x_tensor = torch.tensor(res_test, dtype=torch.float).to(device) #.requires_grad_(True)
    query_y_tensor = torch.tensor(data_adult_test_labels, dtype=torch.long).to(device)
    
    dataset_train = TensorDataset(train_x_tensor, train_y_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=128)
    
    with torch.no_grad():
        clf.fit(input_x_tensor.detach(), input_y_tensor.flatten().detach())
        predictions = clf.predict_proba(query_x_tensor)
        res_accuracy = accuracy_score(data_adult_test_labels, (predictions[:,1].detach().cpu()>0.5))
        print("Initial:", res_accuracy)

    input_x_tensor.requires_grad_(True)
    input_y_tensor.requires_grad_(True)
    
    optim_impl = Adam([input_x_tensor, input_y_tensor], lr=5e-3)
    loss_batches = []
    acc_batches = []
    steps = 0
    for data_batch in tqdm(dataloader_train):
        input_x_batch, input_y_batch = data_batch
        optim_impl.zero_grad()
        clf.fit(input_x_tensor, input_y_tensor.flatten())
        predictions = clf.predict_proba(train_x_tensor)
        lossfn = torch.nn.NLLLoss()
        loss = lossfn(torch.log(predictions), train_y_tensor)
        loss.backward()
        optim_impl.step()
        if steps % 5 == 0:
            with torch.no_grad():
                predictions = clf.predict_proba(query_x_tensor)
                res_accuracy = accuracy_score(data_adult_test_labels, (predictions[:,1].detach().cpu()>0.5))
                print("Test acc:", res_accuracy)
                print("Test Loss:", loss.item())
                loss_batches.append(loss.item())
                acc_batches.append(res_accuracy)
        steps += 1
        json.dump({"loss": loss_batches, "acc": acc_batches}, open("prompttune.json", "w"))