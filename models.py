from ModelHeaders import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import preprocessing as pp
from preprocessing import TextDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from constants import LR_BIAS_THRESHOLD, BATCH_SIZE, NUM_EPOCHS, SEED, MAX_EPOCHS, AUTO_LOSS_THRESHOLD
from statistics import plot_metrics


def test_model(model, data, emb_column, use_torch: bool = False, biased_pred: bool = False,
               printouts: bool = False) -> (float, float, float):
    """
    Used to test a single model on a single dataset.
    :param model: model to test.
    :param data: data to use.
    :param emb_column: name of the column with the embedding.
    :param use_torch: whether the model is torch based.
    :param biased_pred: whether the model has biased prediction defined.
    :param printouts: whether to print out the results.
    :return: accuracy, precision, recall of the model.
    """
    # setting the seed for reproducibility
    torch.manual_seed(SEED)

    train_data, test_data = pp.split_data(data, 0.7)
    X_train = np.stack(train_data[emb_column].values, axis=0)
    y_train = train_data['interest'].to_numpy(dtype=np.int32)

    X_test = np.stack(test_data[emb_column].values, axis=0)
    y_test = test_data['interest'].to_numpy(dtype=np.int32)

    if not use_torch:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        bias_predictions = model.biased_predict(X_test, threshold=LR_BIAS_THRESHOLD)
    else:
        dataset = TextDataset(train_data, emb_column)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_dataset = TextDataset(test_data, emb_column)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = train_torch_model(model, dataloader, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, test_accuracy=True,
                                  test_dataloader=test_dataloader, printouts=printouts)
        predictions = model(to_device(torch.tensor(X_test, dtype=torch.float32))).detach().cpu().numpy()
        predictions = model.output_to_labels(predictions)

    accuracy, precision, recall, _ = evaluate_model(y_test, predictions)
    if printouts:
        print('Regular predictions:')
        print(f'Accuracy: {accuracy:.2f},\nPrecision: {precision:.2f},\nRecall: {recall:.2f}')

    if biased_pred:
        accuracy_bias, precision_bias, recall_bias, _ = evaluate_model(y_test, bias_predictions)
        if printouts:
            print('Biased predictions:')
            print(f'Accuracy: {accuracy_bias:.2f},\nPrecision: {precision_bias:.2f},\nRecall: {recall_bias:.2f}')

        return accuracy_bias, precision_bias, recall_bias

    return accuracy, precision, recall


def test_folded_model(model, data, emb_column, folds=5, metrics: str = 'f1', printouts: bool = False) -> tuple or float:
    """
    This function makes sense only for torch models, therefore is it not backward compatible.
    Used to train the given model on the data, while the training is done on the folds.
    :param model: model to test.
    :param data: data to use.
    :param emb_column: column to use for the embeddings.
    :param folds: number of folds.
    :param metrics: which metrics to use for evaluation. Is either 'accuracy', 'precision', 'recall' or 'f1', or 'all'.
    :return: accuracy, precision, recall of the model. Or an individual metric if specified.
    """

    train_data, test_data = pp.split_data(data, 0.8)
    folded_data = pp.fold_split(train_data, folds)

    for fold in tqdm(range(folds), total=folds, desc='Training the model on the folds', disable=not printouts):
        train_data = pd.concat([folded_data[i] for i in range(folds) if i != fold])

        dataset = TextDataset(train_data, emb_column)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = train_torch_model(model, dataloader, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, printouts=False)

    X_test = np.stack(test_data[emb_column].values, axis=0)
    y_test = test_data['interest'].to_numpy(dtype=np.int64)
    # testing the model
    predictions = model(to_device(torch.tensor(X_test, dtype=torch.float32))).detach().cpu().numpy()
    predictions = model.output_to_labels(predictions)

    accuracy, precision, recall, f1 = evaluate_model(y_test, predictions)
    if printouts:
        print(f'Accuracy: {accuracy:.2f},\nPrecision: {precision:.2f},\nRecall: {recall:.2f}')

    if metrics == 'accuracy':
        return accuracy
    elif metrics == 'precision':
        return precision
    elif metrics == 'recall':
        return recall
    elif metrics == 'f1':
        return f1

    return accuracy, precision, recall, f1

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
    f1 = f1_score(ground_truth, predictions, zero_division=0)

    return accuracy, precision, recall, f1


def cross_validation(data, emb_column, models, model_names, folds=5, use_torch=False):
    """
    Perform cross validation on the data using the models.
    :param data: the data to use.
    :param emb_column: the column to use for the embeddings.
    :param models: the models to use.
    :param model_names: the names of the models.
    :param folds: the number of folds.
    :param use_torch: whether the model is a torch model.
    :return: the results of the cross validation.
    """

    folded_data = pp.fold_split(data, folds)

    for (model, model_name) in zip(models, model_names):
        accuracies = []
        precisions = []
        recalls = []

        print(f'--- Training the {model_name} model ---')
        for fold in tqdm(range(folds), total=folds, desc=f'Cross validation for {model_name} model'):
            # if it is a torch model, we need to set it fresh for each fold
            if use_torch:
                model = model.__class__()

            train_data = pd.concat([folded_data[i] for i in range(folds) if i != fold])
            test_data = folded_data[fold]

            # --- creating the train and test data for the model ---
            # creating a numpy matrix from a list of numpy arrays
            X_train = np.stack(train_data[emb_column].values, axis=0)
            y_train = train_data['interest'].to_numpy(dtype=np.int64)

            X_test = np.stack(test_data[emb_column].values, axis=0)
            y_test = test_data['interest'].to_numpy(dtype=np.int64)

            # --- training the model ---
            if not use_torch:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            else:
                dataset = TextDataset(train_data, emb_column)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

                model = train_torch_model(model, dataloader, batch_size=BATCH_SIZE, epochs='auto', printouts=False)
                predictions = model(to_device(torch.tensor(X_test, dtype=torch.float32))).detach().cpu().numpy()
                predictions = model.output_to_labels(predictions)

            accuracy, precision, recall, _ = evaluate_model(y_test, predictions)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

        print(f'--- {model_name} model trained and tested ---')
        # Plotting box plots for each of the metrics, on the same plot
        plot_metrics(metrics=[accuracies, precisions, recalls], metrics_names=['Accuracy', 'Precision', 'Recall'],
                     model_name=model_name, emb_name=emb_column)


def to_device(x):
    return x.to(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))


def train_torch_model(model: nn.Module, dataloader,
                      optimizer: torch.optim.Optimizer = None, batch_size: int = 16, epochs: int or str = 10,
                      test_accuracy: bool = False, printouts: bool = False, **kwargs) -> nn.Module:
    """
    Used to train the model
    :param model: model to train
    :param dataloader: dataloader to use
    :param optimizer: which optimizer to use, default is Adam
    :param batch_size: size of the batch
    :param epochs: number of epochs to train for, either a number, or 'auto' for automatic training
    :param test_accuracy: whether to test the accuracy, if yes, one more argument is expected: test_dataloader
    :param printouts: whether to print out the progress
    :return: trained model
    """
    if epochs == 'auto':
        epochs = MAX_EPOCHS

    # setting the seed for reproducibility
    torch.manual_seed(SEED)

    if test_accuracy:
        if 'test_dataloader' not in kwargs:
            raise ValueError('If test_accuracy is True, then test_dataloader should be provided.')
        test_dataloader = kwargs['test_dataloader']

    # setting the optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    to_device(model)

    losses = []
    precisions = []
    recalls = []

    inital_loss = None

    for epoch in range(epochs):
        model.train(True)
        if printouts:
            print(f'Epoch {epoch + 1}')

        # To calculate the loss for each epoch
        total_loss = 0

        for i, data in enumerate(dataloader):
            # Forward pass
            inputs, labels = data

            inputs = to_device(inputs)
            labels = to_device(labels)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).view(-1)
            loss = model.loss(outputs, labels)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

        current_loss = total_loss / batch_size
        losses.append(current_loss)
        if printouts:
            print(f'Loss: {current_loss:.4f}')

        if test_accuracy:
            model.eval()
            ground_truth = []
            predictions = []

            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data
                    inputs = to_device(inputs)
                    labels = to_device(labels)
                    ground_truth.extend(labels.cpu().numpy())

                    outputs = model(inputs).view(-1)
                    predictions.extend(model.output_to_labels(outputs.cpu().numpy()))

            accuracy, precision, recall, f1 = evaluate_model(ground_truth, predictions)

            precisions.append(precision)
            recalls.append(recall)

        # if the loss is less than the threshold, then we can stop the training
        if inital_loss is None:
            inital_loss = current_loss
        elif current_loss < inital_loss / AUTO_LOSS_THRESHOLD:
            if printouts:
                print(f'Loss is less than {AUTO_LOSS_THRESHOLD} times the initial loss, stopping the training.')
                print(f'Initial loss: {inital_loss:.4f}, Current loss: {current_loss:.4f}')
                print(f'Epochs trained: {epoch + 1}')
            break

    # after training, in case of testing accuracy, plotting the metrics
    if test_accuracy:
        plt.figure()
        plt.plot(losses, label='Loss')
        plt.plot(precisions, label='Precision')
        plt.plot(recalls, label='Recall')
        plt.title('Losses, Precision and Recall over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'plots/{model.name}.png')

    model.train(False)
    return model


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
