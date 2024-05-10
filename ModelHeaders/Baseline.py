class Baseline:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [1] * len(X)