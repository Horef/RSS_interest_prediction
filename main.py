import numpy as np
import pandas as pd

import preprocessing as pp
import embedding as emb
from constants import TRAIN_PATH
from ModelHeaders.LogisticLinearRegression import LogisticLinearRegression
from ModelHeaders.SoftSVM import SoftSVM
from ModelHeaders.KNN import KNN
from models import test_model, cross_validation



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

    simple_data_emb = emb.simple_embedding(data, w2v_stanford, w2v_local, agg='mean')
    stacked_data_emb = emb.stacked_embedding(data, w2v_stanford, w2v_local, agg='mean')

    # --- training and testing different models ---
    models_classes = [LogisticLinearRegression(), SoftSVM(), KNN()]
    models_names = ['Logistic Linear Regression', 'Soft SVM', 'KNN']
    cross_validation(simple_data_emb, 'simple_embedding', models_classes, models_names, k=5)