import numpy as np
import pandas as pd


class MRMRFeatureSelector:
    def __init__(self, k=100, criterion="MID", eps=1e-12):
        self.k = k
        self.criterion = criterion.upper()
        self.eps = eps
        self.selected_indices_ = []
        self.selected_features_ = []
        self.history_ = []
        self.relevance_ = None
        self.redundancy_matrix_ = None

    def forward(self, X, k=None):
        k = k or self.k
        indices = self.selected_indices_[:k]
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, indices]
        return np.asarray(X)[:, indices]

    def __call__(self, X, k=None):
        return self.forward(X, k=k)

    def get_history_df(self, k=None):
        k = k or len(self.history_)
        return pd.DataFrame(self.history_[:k])
