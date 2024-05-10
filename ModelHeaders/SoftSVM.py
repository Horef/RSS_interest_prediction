from sklearn.svm import SVC
from constants import SEED

# Soft SVM model implementation
class SoftSVM:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, class_weight='balanced',
                 probability=True, random_state=SEED):
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0, class_weight=class_weight,
                         probability=probability, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def biased_predict(self, X, threshold=0.5):
        return self.model.predict_proba(X)[:, 1] > threshold

    def proba(self, X):
        return self.model.predict_proba(X)
