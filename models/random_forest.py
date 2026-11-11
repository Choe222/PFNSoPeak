from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators=300, max_depth=None, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def forward(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.forward(X)
