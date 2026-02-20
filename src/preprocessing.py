import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class KFoldTargetEncoder:
    def __init__(self, col, n_splits=5):
        self.col = col
        self.n_splits = n_splits
        self.global_mean = None
        self.target_means = None

    def fit_transform(self, X, y):
        X = X.copy()
        self.global_mean = y.mean()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        encoded_col = pd.Series(index=X.index, dtype=float)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            means = y_train.groupby(X_train[self.col]).mean()

            encoded_col.iloc[val_idx] = X_val[self.col].map(means)

        encoded_col.fillna(self.global_mean, inplace=True)
        X[self.col + "_te"] = encoded_col

        return X.drop(columns=[self.col])

    def transform(self, X):
        X = X.copy()

        X[self.col + "_te"] = X[self.col].map(self.target_means)

        X[self.col + "_te"].fillna(self.global_mean, inplace=True)

        return X.drop(columns=[self.col])

    def fit(self, X, y):
        self.global_mean = y.mean()
        self.target_means = y.groupby(X[self.col]).mean()