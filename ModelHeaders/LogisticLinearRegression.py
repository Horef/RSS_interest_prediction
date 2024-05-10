from sklearn.linear_model import LogisticRegression
from constants import SEED

# A simple logistic regression model, basically a wrapper around the sklearn logistic regression model
class LogisticLinearRegression:
    def __init__(self, class_weight='balanced', max_iter=5000, n_jobs=-1, random_state=SEED):
        # class_weight is used to balance the classes, as there are more 0s than 1s and 1s are more important
        self.model = LogisticRegression(class_weight=class_weight, max_iter=max_iter, n_jobs=n_jobs, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def biased_predict(self, X, threshold=0.5):
        return self.model.predict_proba(X)[:, 1] > threshold

    def predict_proba(self, X):
        return self.model.predict_proba(X)
