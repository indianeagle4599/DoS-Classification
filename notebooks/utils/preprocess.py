import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1337)


def sequential_split(data, test_size=0.25):
    """
    Splits data sequentially based on given dataset and test size

    Args:
        data (pandas.DataFrame): A Pandas DataFrame containing the entire dataset
        test_size (float, optional): Proportion of the data to be used as test data and is always a value between 0 and 1. Defaults to 0.25.

    Returns:
        (train_df,test_df): A tuple containing two Pandas DataFrames, a train and a test set with the same dimensionality as the given DataFrame
    """
    train_len = int(len(data) * (1 - test_size))
    train_df = data.iloc[:train_len, :]
    test_df = data.iloc[train_len:, :]
    return train_df, test_df


def label_encode(df, col):
    class_to_idx = {d: i for i, d in enumerate(df[col].unique())}
    df[col] = df[col].replace(class_to_idx)
    return df


def normalize_data(data, target_col):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop(target_col, axis=1))
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data[target_col] = data[target_col]
    return scaled_data


def preprocess(data):
    data = label_encode(data, 1)
    data = label_encode(data, 2)
    data = label_encode(data, 3)
    data["Y"] = data.loc[:, 41]
    data.drop(columns=[41], inplace=True)
    data = normalize_data(data, "Y")
    data = label_encode(data, "Y")
    train_df, test_df = sequential_split(data)
    return train_df, test_df
