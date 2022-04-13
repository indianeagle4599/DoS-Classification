import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1337)


def sequential_split(X, Y, test_size=0.25):
    train_len = int(len(X) * (1 - test_size))
    X_train = X.iloc[:train_len, :]
    X_test = X.iloc[train_len:, :]
    y_train = Y.iloc[:train_len, :]
    y_test = Y.iloc[train_len:, :]
    return X_train, X_test, y_train, y_test

def xy_split(data):
    X = data.drop(columns=['Y'])
    Y = pd.get_dummies(data['Y'])
    return X, Y


def normalize_data(data, target_col):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(target_col, axis=1))
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data[target_col] = data[target_col]
    return scaled_data


def preprocess(data):
    data = pd.get_dummies(data, columns=[1, 2, 3])
    data["Y"] = data.loc[:, 41]
    data.drop(columns=[41], inplace=True)
    data = normalize_data(data, "Y")
    X, Y = xy_split(data)
    return sequential_split(X, Y)
