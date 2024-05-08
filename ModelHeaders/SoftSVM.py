from sklearn.svm import SVC
from constants import SEED

# Soft SVM model implementation
class SoftSVM:
    def __init__(self):
        self.model = SVC(class_weight='balanced', probability=True, random_state=SEED)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def biased_predict(self, X, threshold=0.5):
        return self.model.predict_proba(X)[:, 1] > threshold

    def proba(self, X):
        return self.model.predict_proba(X)
