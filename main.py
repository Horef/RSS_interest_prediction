import numpy as np
import pandas as pd

import preprocessing as pp
import embedding as emb
from constants import TRAIN_PATH
from ModelHeaders.LogisticLinearRegression import LogisticLinearRegression
from ModelHeaders.SoftSVM import SoftSVM
from ModelHeaders.KNN import KNN
from models import test_model



if __name__ == '__main__':
    print('--- Starting the main script ---')
    # --- loading the pre-trained word2vec models ---
    # loading the stanford w2v model
    w2v_stanford = emb.load_stanford_w2v()

    # loading the local w2v model
    new_data = pp.read_file(TRAIN_PATH)
    w2v_local = emb.train_w2v_model(new_data['title'].values)

    # --- loading the train data ---
    data = pp.load_train_data(TRAIN_PATH)
    data = pp.clean_data(data)

    # printing statistics of the data for the amount of 0s and 1s in the interest column
    data_stats = data['interest'].value_counts()
    print(f'The ratio of 1s in the whole data: {data_stats[1] / data_stats[0]:.2f}')
    print(f'Total length of the data: {len(data)}\n')

    # splitting the data into train and validation such that the representation of the interest
    # column is the same in both
    train_data, val_data = pp.split_data(data)

    # printing the ration of 1s to 0s in the train and test data
    train_stats = train_data['interest'].value_counts()
    val_stats = val_data['interest'].value_counts()
    print(f'The ratio of 1s in the train data: {train_stats[1] / train_stats[0]:.2f}')
    print(f'Total length of the train data: {len(train_data)}\n')
    print(f'The ratio of 1s in the validation data: {val_stats[1] / val_stats[0]:.2f}')
    print(f'Total length of the validation data: {len(val_data)}')

    # --- creating embeddings for the data ---
    train_data_emb = emb.simple_embedding(train_data, w2v_stanford, w2v_local, agg='mean')
    val_data_emb = emb.simple_embedding(val_data, w2v_stanford, w2v_local, agg='mean')



    # --- training the LR model ---
    print(f'--- Training the Simple Logistic Regression model ---')
    model = LogisticLinearRegression()
    test_model(model, X_train=X_train, y_train=y_train,
               X_test=X_test, y_test=y_test)
    print(f'--- Simple Logistic Regression model trained and tested ---')

    # --- training the SVM model ---
    print(f'--- Training the Simple SVM model ---')
    model = SoftSVM()
    test_model(model, X_train=X_train, y_train=y_train,
               X_test=X_test, y_test=y_test)
    print(f'--- Simple SVM model trained and tested ---')

    # --- training the KNN model ---
    print(f'--- Training the Simple KNN model ---')
    model = KNN(n_neighbors=5)
    test_model(model, X_train=X_train, y_train=y_train,
               X_test=X_test, y_test=y_test)
    print(f'--- Simple KNN model trained and tested ---')