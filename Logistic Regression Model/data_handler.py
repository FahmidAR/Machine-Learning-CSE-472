import numpy as np
import pandas as pd


def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    bank_dataset = pd.read_csv('data_banknote_authentication.csv')
    X = bank_dataset.iloc[:, :-1].values
    y = bank_dataset.iloc[:, -1].values

    print(bank_dataset.head())
    print("*"*50+"\n")
    print(bank_dataset.describe())
    print("*"*50+"\n")   
    print(bank_dataset['isoriginal'].value_counts())
    print("*"*50+"\n")

    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    print("*"*50+"\n")
    
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.

    if shuffle:
        shuffled = np.arange(len(X))
        np.random.shuffle(shuffled)
        X_suffled = X[shuffled]
        y_suffled = y[shuffled]

    else:
        X_suffled = X
        y_suffled = y
        
    X_train, y_train, X_test, y_test = X_suffled[:int(len(X) * (1 - test_size))], y_suffled[:int(len(y) * (1 - test_size))], X_suffled[int(len(X) * (1 - test_size)):], y_suffled[int(len(y) * (1 - test_size)):]
   
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print("*"*50+"\n")
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    print("*"*50+"\n")

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement

    index = np.random.choice(len(X), len(X), replace=True)
    X_sample, y_sample = X[index] , y[index]
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
