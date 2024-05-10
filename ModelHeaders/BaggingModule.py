from constants import SEED
import numpy as np
import pandas as pd

class BaggingModule:
    def __init__(self, model, bag_size: int = 5, progressive_seed: bool = False, **kwargs):
        """
        Used to initialize the bagging module
        :param model: model to bag
        :param bag_size: size of the bag
        :param progressive_seed: whether to change the seed for each model
        :param kwargs: arguments to pass to the model
        """
        self.bag = []
        self.bag_size = bag_size

        for i in range(bag_size):
            local_seed = SEED if not progressive_seed else SEED + i
            self.bag.append(model(random_state=local_seed, **kwargs))


    def fit(self, X: np.array, y: np.array):
        """
        Used to fit the bagged models
        :param X: input data
        :param y: target data
        """
        local_seed = SEED

        bootstrap_samples = []
        for i in range(self.bag_size):
            local_seed += i

            while True:
                # setting the seed for reproducibility
                rng = np.random.default_rng(local_seed)

                # finding the indices for the bootstrap sample
                indices = rng.choice(range(len(X)), self.bag_size, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]

                if sum(y_sample) > 0:
                    break
                else:
                    local_seed += 1

            bootstrap_samples.append((X_sample, y_sample))

        for model, sample in zip(self.bag, bootstrap_samples):
            model.fit(sample[0], sample[1])

    def predict(self, X):
        """
        Used to predict with the bagged models
        :param X: input data
        :return: predictions
        """
        predictions = []
        for model in self.bag:
            predictions.append(model.predict(X))

        prediction_mean = np.mean(predictions, axis=0)
        prediction = np.where(prediction_mean >= 0.5, 1, 0)

        return prediction