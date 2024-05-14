from ModelHeaders import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

import preprocessing as pp
from sklearn.metrics import accuracy_score, precision_score, recall_score
from constants import LR_BIAS_THRESHOLD


def test_model(model, X_train, y_train, X_test, y_test) -> (float, float, float):
    """
    Used to test the model.
    :param model: model to test.
    :param X_train: training data.
    :param y_train: training labels.
    :param X_test: testing data.
    :param y_test: testing labels.
    :return: accuracy, precision, recall of the model.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    bias_predictions = model.biased_predict(X_test, threshold=LR_BIAS_THRESHOLD)

    # if all predictions are 0, then the model is not working properly
    if sum(predictions) == 0:
        print('Model is not working properly for regular predictions, all predictions are 0.')
    else:
        print('Regular predictions:')
        evaluate_model(y_test, predictions)

    if sum(bias_predictions) == 0:
        print('Model is not working properly for biased predictions, all predictions are 0.')
        return 0, 0, 0
    else:
        print('Biased predictions:')
        return evaluate_model(y_test, bias_predictions)


def prepare_performance_model(model, data, emb_column):
    """
    Used to prepare the model for performance use.
    :param model: model to prepare.
    :param data: data to use.
    :param emb_column: column to use for the embeddings.
    :return: the model.
    """
    X_train = np.stack(data[emb_column].values, axis=0)
    y_train = data['interest'].to_numpy(dtype=np.int64)

    model.fit(X_train, y_train)
    return model

def evaluate_model(ground_truth: [int], predictions: [int]) -> (float, float, float):
    """
    Evaluates the model based on the ground truth and the predictions.
    :param ground_truth: the ground truth.
    :param predictions: the predictions.
    :return: accuracy, precision, recall of the model.
    """
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)

    return accuracy, precision, recall


def cross_validation(data, emb_column, models, model_names, folds=5):
    """
    Perform cross validation on the data using the models.
    :param data: the data to use.
    :param emb_column: the column to use for the embeddings.
    :param models: the models to use.
    :param model_names: the names of the models.
    :param folds: the number of folds.
    :return: the results of the cross validation.
    """

    folded_data = pp.fold_split(data, folds)

    for (model, model_name) in zip(models, model_names):
        accuracies = []
        precisions = []
        recalls = []

        print(f'--- Training the {model_name} model ---')
        for fold in range(folds):
            train_data = pd.concat([folded_data[i] for i in range(folds) if i != fold])
            test_data = folded_data[fold]

            # --- creating the train and test data for the model ---
            # creating a numpy matrix from a list of numpy arrays
            X_train = np.stack(train_data[emb_column].values, axis=0)
            y_train = train_data['interest'].to_numpy(dtype=np.int64)

            X_test = np.stack(test_data[emb_column].values, axis=0)
            y_test = test_data['interest'].to_numpy(dtype=np.int64)

            # --- training the model ---
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy, precision, recall = evaluate_model(y_test, predictions)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

        print(f'--- {model_name} model trained and tested ---')
        # Plotting box plots for each of the metrics, on the same plot
        plot_metrics(accuracies, precisions, recalls, model_name, emb_name=emb_column)


def plot_metrics(accuracies, precisions, recalls, model_name, emb_name):
    """
    Plot the metrics for the models.
    :param accuracies: the accuracies.
    :param precisions: the precisions.
    :param recalls: the recalls.
    :param model_name: the names of the model.
    :param emb_name: name of the embedding used
    """
    labels = ['Accuracy', 'Precision', 'Recall']

    # initializing a new plot
    plt.figure()
    plt.boxplot([accuracies, precisions, recalls], labels=labels)
    plt.title(f'Model Metrics for {model_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.savefig(f'plots/{model_name}_w_{emb_name}_metrics.png')


def save_model(model, model_name):
    """
    Save the model to a file.
    :param model: the model to save.
    :param model_name: the name of the model.
    """
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)


def save_models(models, model_names):
    """
    Save the models to files.
    :param models: the models to save.
    :param model_names: the names of the models.
    """
    for model, model_name in zip(models, model_names):
        save_model(model, model_name)


def load_model(model_name):
    """
    Load the model from a file.
    :param model_name: the name of the model.
    :return: the model.
    """
    # joining the path to the model with the full path
    model_path = os.path.join(os.path.dirname(__file__), f'models/{model_name}.pkl')

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_models(model_names):
    """
    Load the models from files.
    :param model_names: the names of the models.
    :return: the models.
    """
    return [load_model(model_name) for model_name in model_names]
