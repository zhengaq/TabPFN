"""An example for finetuning TabPFN on the California Housing Regression dataset."""
import json
from functools import partial
from pathlib import Path
import numpy as np

import sklearn
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

#from tabpfn.regressor import TabPFNRegressor
from tabpfn import TabPFNRegressor
from tabpfn.utils import collate_for_tabpfn_dataset
from tabpfn.base import RegressorModelSpecs

import copy



def eval_test_regression_standard(reg: TabPFNRegressor, 
                                  X_train_raw: np.ndarray, y_train_raw: np.ndarray,
                                  X_test_raw: np.ndarray, y_test_raw: np.ndarray):
    if hasattr(reg, 'model_') and reg.model_ is not None:
        #My eval was manipulating the underlying reg class
        new_model = copy.deepcopy(reg.model_) # <--- Need!!!
        new_config = copy.deepcopy(reg.config_) #probs do not need
        new_bar_dist = copy.deepcopy(reg.bardist_) #probs do not need
        model_spec_obj = RegressorModelSpecs(
            model=new_model,
            config=new_config,
            norm_criterion=new_bar_dist,
        )
        reg_eval = TabPFNRegressor(model_path=model_spec_obj)

        #reg_eval.memory_saving_mode = False
    else: 
        print("Pre-Trained Model Performance")
        reg_eval = TabPFNRegressor()

    reg_eval.fit(X_train_raw, y_train_raw)
    #if hasattr(reg, 'model_') and reg.model_ is not None:
        #print("reg paramters: ", [p for p in reg.model_.parameters()][0][10])
        #print("reg_eval paramters: ", [p for p in reg_eval.model_.parameters()][0][10])

    predictions = reg_eval.predict(X_train_raw) # Default output is 'mean'

    print(f"Train MSE: {mean_squared_error(y_train_raw, predictions)}")
    print(f"Train MAE: {mean_absolute_error(y_train_raw, predictions)}")
    print(f"Train R2: {r2_score(y_train_raw, predictions)}")
    
    try:
        predictions = reg_eval.predict(X_test_raw) # Default output is 'mean'
        #print("predictions", predictions[0])
        mse = mean_squared_error(y_test_raw, predictions)
        mae = mean_absolute_error(y_test_raw, predictions)
        r2 = r2_score(y_test_raw, predictions)
    except Exception as e:
        print(f"Error during evaluation prediction/metric calculation: {e}")
        mse, mae, r2 = np.nan, np.nan, np.nan # Return NaNs on error            

    return mse, mae, r2


if __name__ == "__main__":
    device = "cpu"
    n_use =500
    do_epochs = 10
    
    data_frame_x, data_frame_y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    splitfn = partial(train_test_split, test_size=0.3)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitfn(data_frame_x[:n_use], data_frame_y[:n_use])

    print(X_train_raw.shape)
    print(X_test_raw.shape)


    regressor_args = dict(
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=2,
        random_state=2, # For reproducibility of internal sampling/preprocessing
        inference_precision=torch.float32, # Keep precision consistent
        # memory_saving_mode default is 'auto'
    )

    reg = TabPFNRegressor(**regressor_args, differentiable_input=False)
    
    print("Pretrained Model Performance ")
    
    res_mse, res_mae, res_r2 = eval_test_regression_standard(
                reg, X_train_raw, y_train_raw, X_test_raw, y_test_raw
            )
    
    print(f"Test MSE: {res_mse:.4f}")
    print(f"Test MAE: {res_mae:.4f}") # Added MAE printout
    print(f"Test R2: {res_r2:.4f}")   # Added R2 printout

    

    datasets_collection = reg.get_preprocessed_datasets(X_train_raw, y_train_raw, splitfn, max_data_size=1000)
    datasets_collection_test = reg.get_preprocessed_datasets(X_test_raw, y_test_raw, splitfn, max_data_size=1000)

    my_dl_train = DataLoader(datasets_collection, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(datasets_collection_test, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    optim_impl = Adam(reg.model_.parameters(), lr=1e-5)
    
    loss_batches = []
    mse_batches = []
    
    #TODO: do with actual model
    
    #OK this we become a problem 
    #lossfn = reg.renormalized_criterion_

    for epoch in range(do_epochs):
        #Otherwise I cannot use the .fit function inside is problematic
        #reg.memory_saving_mode = False
        for data_batch in tqdm(my_dl_train):       
                 
            #print("paramter 0 ", [p for p in reg.model_.parameters()][0][10])

            optim_impl.zero_grad()
            
            (X_trains_preprocessed, X_tests_preprocessed, y_trains_preprocessed,
             y_test_standardized, cat_ixs, confs, renormalized_criterion, 
             batch_x_test_raw, batch_y_test_raw) = data_batch

            #chill, uses the same Inference Engine as classification
            reg.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)

            lossfn = reg.renormalized_criterion_

            #Output to logit space: [BatchSize, N_test, NumBars]
            logits_pred = reg.predict_from_preprocessed(X_tests_preprocessed) # torch.Size([105, 1]) #[BatchSize, N_test, NumBars]
            y_target = batch_y_test_raw.to(device) # Shape: [BatchSize, N_test]

            #loss = lossfn(logits_pred, y_test_raw.to(device))
            #nll_loss_per_sample = lossfn(logits_pred, batch_y_test_raw.to(device))
            nll_loss_per_sample = lossfn(logits_pred, y_test_standardized.to(device))

            #print #torch.Size([1, 105]))

            loss = nll_loss_per_sample.mean()

            print(f" Loss in EPOCH {epoch+1}: {loss}")
            loss.backward()
            optim_impl.step()
        
        print(f"---- EPOCH {epoch+1} Evaluation Results (Standard Predict): ----")
        
        
        res_mse, res_mae, res_r2 = np.nan, np.nan, np.nan        
        res_mse, res_mae, res_r2 = eval_test_regression_standard(
                reg, X_train_raw, y_train_raw, X_test_raw, y_test_raw
            )
        
        print(f"Test MSE: {res_mse:.4f}")
        print(f"Test MAE: {res_mae:.4f}") # Added MAE printout
        print(f"Test R2: {res_r2:.4f}")   # Added R2 printout

        

        #loss_batches.append(loss_test)
        #mse_batches.append(res_mse)
        #with Path("finetune_regression.json").open(mode="w") as file:
        #    json.dump({"loss": loss_batches, "acc": mse_batches}, file)

