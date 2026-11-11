from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def forward(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.forward(X)
