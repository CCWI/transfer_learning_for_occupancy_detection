# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

def walk_forward(data, window_size=15):
    '''
    Divides a time series into moving windows of a given window size.
    Windows are moved forward by one time step.
    '''
    data = np.array(data)
    if len(data.shape)==1: # if only one feature -> make multidim array
        data = data.reshape(len(data), 1)
    sequences = []
    for i in range(0, len(data)-window_size+1):
        sequences.append(data[i:i+window_size, :])
    return np.array(sequences)

class SplitMode(Enum):
    NO_SPLIT = 0
    TRAIN_TEST_SPLIT = 1
    TRAIN_VALIDATION_TEST_SPLIT = 2
    
def prepare_data(x, y, x_src=[], y_src=[], window_size=15, max_batch_size=32, splitAt=None, normalize=True):
    '''
    Splits time series data from target domain and source domain (if available) into training and test datasets.
    In case of one or multiple source domains, domain information (target domain (1) or other (0)) is stored as an additional label in y[1]. 
    x                time series data of the target domain
    y                target domain labels
    x_src            data from source domain(s): array of datasets per domain (None if there is only the target domain)
    y_src            labels from source domain(s): array of datasets per domain (None if there is only the target domain)
    window_size      size of each sample sequence
    max_batch_size   maximum batch size that will be used during training (last batch at this size will be ignored if it has not enough samples)
    splitAt          chose one of the following options:
                     - index of the sample at which the data shall be split into train and test data (integer value)
                     - fraction [0, 1) of the available data may be passed (float value)
                     - list of two indices or fractions for splitting into training, validation and test dataset
                     - None for using all data as training data
    '''
    # determine split mode
    if splitAt == None:
        splitMode = SplitMode.NO_SPLIT
    elif type(splitAt) in [float, int]:
        splitMode = SplitMode.TRAIN_TEST_SPLIT
    elif (type(splitAt) == list) and (len(splitAt) == 2):
        splitMode = SplitMode.TRAIN_VALIDATION_TEST_SPLIT
    else:
        raise ValueError("splitAt must either be a number (index or fraction of the data), a list of two numbers (for train-validation-test split), or None")
        
    # ensure that data is organized in numpy arrays
    x = np.array(x)
    y = np.array(y)
    for i in range(0, len(x_src)):
        x_src[i] = np.array(x_src[i])
        y_src[i] = np.array(y_src[i])
            
    # walk forward window generation
    x = walk_forward(x, window_size)
    y = walk_forward(y, window_size)

    for i in range(0, len(x_src)):
        x_src[i] = walk_forward(x_src[i], window_size)
        y_src[i] = walk_forward(y_src[i], window_size)
    
    # train-test split or train-validation-test split (split only in target domain)
    if splitMode == SplitMode.NO_SPLIT:
        print("no train-test split applied")
        x_train = x
        y_train = y
       
    elif splitMode == SplitMode.TRAIN_TEST_SPLIT:
        print("train-test split")
        if splitAt < 1: 
            splitAt = int(round(len(x) * splitAt, 0))
        print("split at index", splitAt)
        x_train = x[0:splitAt]
        x_test = x[splitAt:]    
        y_train = y[0:splitAt]
        y_test = y[splitAt:]
    elif splitMode == SplitMode.TRAIN_VALIDATION_TEST_SPLIT:
        print("train-validation-test split")
        if splitAt[0] < 1: 
            splitAt[0] = int(round(len(x) * splitAt[0], 0))
        if splitAt[1] < 1: 
            splitAt[1] = int(round(len(x) * splitAt[1], 0))
        print("split at", splitAt)
        x_train = x[0:splitAt[0]]
        y_train = y[0:splitAt[0]]
        x_val = x[splitAt[0]:splitAt[1]]
        y_val = y[splitAt[0]:splitAt[1]]
        x_test = x[splitAt[1]:] 
        y_test = y[splitAt[1]:]
        
    # reshape y
    y_train = y_train[:,-1:].reshape(y_train.shape[0], 1)
    print("train:", np.shape(x_train), np.shape(y_train))
    if splitMode.value > 0:
        y_test = y_test[:,-1:].reshape(y_test.shape[0], 1)
        print("train:", np.shape(x_train), np.shape(y_train), "test:", np.shape(x_test), np.shape(y_test))
    if splitMode.value > 1:
        y_val = y_val[:,-1:].reshape(y_val.shape[0], 1)
        print("train:", np.shape(x_train), np.shape(y_train), "validate:", np.shape(x_val), np.shape(y_val), "test:", np.shape(x_test), np.shape(y_test))
    for i in range(0, len(y_src)):
        y_src[i] = y_src[i][:,-1:].reshape(y_src[i].shape[0], 1)
        
    # add source domain data to training data & insert domain labels (only in multi-domain case)
    if x_src != []:
        print(len(x_train), "training samples from target domain")
    else:
        print(len(x_train), "training samples")
    domain_labels = np.array([[0] for j in range(len(y_train))])
    for i in range(0, len(x_src)):
        print(len(x_src[i]), "training samples from source domain", i+1)
        x_train = np.concatenate([x_train, x_src[i]])
        y_train = np.concatenate([y_train, y_src[i]])
        domain_labels = np.concatenate([domain_labels, np.array([[i+1] for j in range(len(y_src[i]))])])      
    domain_labels[domain_labels > 0] = 1 # binary: either target or other domain
    
    # avoid incomplete last batch
    used_samples = x_train.shape[0] - (x_train.shape[0] % max_batch_size)
    x_train = x_train[:used_samples]
    y_train = y_train[:used_samples]
    domain_labels = domain_labels[:used_samples]
    if splitMode.value > 0:
        used_samples = x_test.shape[0] - (x_test.shape[0] % max_batch_size)
        x_test = x_test[:used_samples]
        y_test = y_test[:used_samples]
    if splitMode.value > 1:
        used_samples = x_val.shape[0] - (x_val.shape[0] % max_batch_size)
        x_val = x_val[:used_samples]
        y_val = y_val[:used_samples]
        
    # add domain labels to y (in multidomain case)
    if x_src != [] and y_src != []:
        y_train = y_train, domain_labels # training data from target domain (0) or source domain (1)
        if splitMode.value > 0:   
            y_test = y_test, np.array([[0] for i in range(0, len(y_test))]) # test data only drawn from target domain (0)
        if splitMode.value > 1:
            y_val  = y_val, np.array([[0] for i in range(0, len(y_val))]) # validation data only drawn from target domain (0)
    
    # normalize features to range [0, 1]
    if normalize:
        shapes = [x_train.shape[0], x_train.shape[1], x_train.shape[2]]
        x_train = MinMaxScaler().fit_transform(x_train.reshape(-1, 1))
        x_train = x_train.reshape(shapes[0], shapes[1], shapes[2])
        if splitMode.value > 0:
            shapes = [x_test.shape[0], x_test.shape[1], x_test.shape[2]]
            x_test = MinMaxScaler().fit_transform(x_test.reshape(-1, 1))
            x_test = x_test.reshape(shapes[0], shapes[1], shapes[2])
        if splitMode.value > 1:
            shapes = [x_val.shape[0], x_val.shape[1], x_val.shape[2]]
            x_val = MinMaxScaler().fit_transform(x_val.reshape(-1, 1))
            x_val = x_val.reshape(shapes[0], shapes[1], shapes[2])
        print("data normalized to range [0, 1]")
   
    if splitMode == SplitMode.NO_SPLIT:
        print("train:", np.shape(x_train), np.shape(y_train))
        return x_train, y_train
    elif splitMode == SplitMode.TRAIN_TEST_SPLIT:
        print("train:", np.shape(x_train), np.shape(y_train), "test:", np.shape(x_test), np.shape(y_test))
        return x_train, y_train, x_test, y_test
    elif splitMode == SplitMode.TRAIN_VALIDATION_TEST_SPLIT:
        print("train:", np.shape(x_train), np.shape(y_train), "validate:", np.shape(x_val), np.shape(y_val), "test:", np.shape(x_test), np.shape(y_test))
        return x_train, y_train, x_val, y_val, x_test, y_test