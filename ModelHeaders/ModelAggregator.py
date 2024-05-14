import numpy as np

class ModelAggregator:
    def __init__(self, models: [object], pretrained: bool = True, model_params: [dict] = None):
        self.models = models
        self.pretrained = pretrained
        self.model_params = model_params

        if not self.pretrained:
            defined_models = []
            if self.model_params is None:
                raise ValueError('Model parameters must be provided if not pretrained')
            if len(self.models) != len(self.model_params):
                raise ValueError('Number of models must match the number of model parameters')
            for model, params in zip(self.models, self.model_params):
                model = model(**params)
                defined_models.append(model)
            self.models = defined_models

    def train(self, X, y):
        if self.pretrained:
            raise ValueError('Cannot train pretrained models')
        for model in self.models:
            model.train(X, y)

    def predict(self, X) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        prediction_mean = np.mean(predictions, axis=0)
        prediction = np.where(prediction_mean >= 0.5, 1, 0)
        return prediction