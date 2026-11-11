from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(
        self,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric=eval_metric,
        )

    def forward(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.forward(X)
