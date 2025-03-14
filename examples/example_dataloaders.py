import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
def get_adult_preprocessed_inputs(adult_path ="../tabpfn-eval/datasets"):

    data_adult_train = pd.read_csv(f"{adult_path}/adult.data", header=None)
    data_adult_test = pd.read_csv(f"{adult_path}/adult.test", header=None)
    data_adult_train.drop(2, axis=1)
    data_adult_test.drop(2, axis=1)
    
    data_adult_train = data_adult_train.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data_adult_test = data_adult_test.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data_adult_test_labels = (data_adult_test[14] == ">50K.").astype(int)
    data_adult_train_labels = (data_adult_train[14] == ">50K").astype(int)
    
    # Print label distribution
    print(data_adult_test_labels.mean(), data_adult_train_labels.mean())
    
    # Drop labels
    data_adult_train.drop(14, axis=1)
    data_adult_test.drop(14, axis=1)
    
    enc = OrdinalEncoder()

    cat_features = [1, 3, 5, 6, 7, 8, 9, 13]
    num_features = [0, 4, 10, 11, 12]
    res = enc.fit_transform(data_adult_train[cat_features])
    res = np.concatenate((res, data_adult_train[num_features].values), axis=1)
    res_test = enc.transform(data_adult_test[cat_features])
    res_test = np.concatenate((res_test, data_adult_test[num_features].values), axis=1)
    return res, data_adult_train_labels, res_test, data_adult_test_labels, [0, 1, 2, 3, 4, 5, 6, 7]