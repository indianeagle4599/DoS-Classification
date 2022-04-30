from cgi import test
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report

TEST_SIZE = 0.3


def get_classification_report(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)
    return classification_report(y_test, y_pred_bool)


np.random.seed(1337)


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
    train_df, test_df = train_test_split(data, test_size=TEST_SIZE, stratify=data["Y"])
    return train_df, test_df


def encode(data):
    data = label_encode(data, 1)
    data = label_encode(data, 2)
    data = label_encode(data, 3)
    data["Y"] = data.loc[:, 41]
    data.drop(columns=[41], inplace=True)
    data = label_encode(data, "Y")
    return data


def train_and_evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=4,
    batch_size=1024,
    validation_split=0.25,
):
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    print("Evaluating Model: ")
    model.evaluate(X_test, y_test, batch_size)
    report = get_classification_report(X_test, y_test, model)
    return history, report


def load_and_evaluate_model(
    model_name,
    X_test,
    y_test,
    batch_size=1024,
):
    model = load_model('../models/'+model_name)
    print(f"Evaluating {model_name.split('.')[0]}: ")
    model.evaluate(X_test, y_test, batch_size)
    report = get_classification_report(X_test, y_test, model)
    print(report)
    return report
