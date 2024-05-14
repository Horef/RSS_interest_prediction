import urllib.error

import pandas as pd

from constants import STANFORD_W2V_PATH, LOCAL_W2V_PATH, SEED

from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import numpy as np
import csv
import urllib.request as request
from zipfile import ZipFile
import os


def load_stanford_w2v(printouts: bool = True) -> KeyedVectors:
    """
    Used to load the pre-trained glove model from the Stanford website.
    :param printouts: whether to print out the progress of the function
    :return: pre-trained glove model
    """
    # loading the glove model from the glove.840B.300d.txt file, if it exists
    # if not, download the glove model
    if printouts:
        print('--- Loading the Stanford w2v model ---')

    try:
        if printouts:
            print('Trying to load the Stanford w2v model')
        w2v_model = KeyedVectors.load(os.path.join(os.path.dirname(__file__), STANFORD_W2V_PATH))
    except FileNotFoundError:
        if printouts:
            print('w2v model not found, downloading stanford glove model')

        url = 'https://nlp.stanford.edu/data/glove.840B.300d.zip'
        filename = 'glove.840B.300d.zip'

        # if the file is already downloaded, skip the download
        if not os.path.exists(os.path.join(os.path.dirname(__file__), filename)):
            try:
                request.urlretrieve(url, filename)
            except urllib.error.URLError:
                if printouts:
                    print('Main link is down, trying the mirror link')
                try:
                    mirror = 'https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'
                    request.urlretrieve(mirror, filename)
                except urllib.error.URLError:
                    if printouts:
                        print('Both links are down, trying a third link')
                    try:
                        mirror = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'
                        request.urlretrieve(mirror, filename)
                    except urllib.error.URLError:
                        raise ConnectionError('All links are down, please try again later')
        else:
            if printouts:
                print('Glove model already downloaded. Skipping the download step')

        if printouts:
            print('Glove model downloaded, extracting the files')
        # if the file is already extracted, skip the extraction
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt')):
            with ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(__file__))
        else:
            if printouts:
                print('Glove model already extracted. Skipping the extraction step')

        if printouts:
            print('Converting the glove model to word2vec format')
        glove_file = datapath(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt'))
        word2vec_glove_file = get_tmpfile("glove.840B.300d.word2vec.txt")
        num_vec, dim_vec = glove2word2vec(glove_file, word2vec_glove_file)
        w2v_model = KeyedVectors.load_word2vec_format(word2vec_glove_file, limit=num_vec - 1)
        w2v_model.save(STANFORD_W2V_PATH)

        if printouts:
            print('Cleaning up the files')
        os.remove(os.path.join(os.path.dirname(__file__), filename))
        os.remove(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt'))

    if printouts:
        print('--- Stanford w2v model loaded ---\n')
    return w2v_model


def train_local_w2v_model(sentences: [[str]], vector_size: int = 300, window: int = 2, min_count: int = 1, workers: int = 12,
                          epochs: int = 100, printouts: bool = True) -> Word2Vec:
    """
    Used to train a word2vec model on a given corpus.
    :param sentences: corpus to train the model on.
    :param vector_size: desired size of the word vectors.
    :param window: window size.
    :param min_count: minimum count of a word to be included.
    :param workers: number of workers to use.
    :param epochs: number of epochs.
    :param printouts: whether to print out the progress of the function
    :return: trained word2vec model.
    """
    if printouts:
        print('--- Training (or updating) local word2vec model ---')
    # if there is a pre-trained model, load it and update it
    if os.path.exists(LOCAL_W2V_PATH):
        if printouts:
            print('Loading the pre-trained word2vec model')
        w2v_model = Word2Vec.load(LOCAL_W2V_PATH)
        w2v_model.build_vocab(sentences, update=True)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epochs)
        w2v_model.save(LOCAL_W2V_PATH)
        if printouts:
            print('--- Word2Vec model updated and saved ---\n')
        return w2v_model
    else:
        if printouts:
            print('No pre-trained word2vec model found')
            print('Training a new word2vec model')
        # Initializing the word2vec model
        w2v_model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                             min_count=min_count, workers=workers, epochs=epochs,
                             seed=SEED)
        w2v_model.save(LOCAL_W2V_PATH)
        if printouts:
            print('--- Word2Vec model trained and saved ---\n')
        return w2v_model


def get_local_w2v_model(printouts: bool = True) -> Word2Vec:
    """
    Used to load the local word2vec model without training it.
    Can be used in the final prediction step.
    :param printouts: whether to print out the progress of the function
    :return: local word2vec model
    """
    if printouts:
        print('--- Loading the local word2vec model ---')

    w2v_path = os.path.join(os.path.dirname(__file__), LOCAL_W2V_PATH)
    # in case the model is not found resulting in an error
    if not os.path.exists(w2v_path):
        raise FileNotFoundError('Local word2vec model not found. Please train the model first.')

    w2v_model = Word2Vec.load(w2v_path)
    if printouts:
        print('--- Local word2vec model loaded ---\n')
    return w2v_model


def simple_embedding(data: pd.DataFrame, w2v_stanford: KeyedVectors, w2v_local: Word2Vec,
                     agg: str = 'sum', printouts: bool = True) -> pd.DataFrame:
    """
    Used to create simple embeddings for the sentences in the data.
    Each sentence is encoded as a single vector, which is the aggregation of the word vectors in the sentence.
    :param data: data to create embeddings for
    :param w2v_stanford: pre-trained word2vec model from Stanford
    :param w2v_local: pre-trained word2vec model from the local data
    :param agg: aggregation method to use for the embeddings. Either 'sum' or 'mean', default is 'sum'
    :param printouts: whether to print out the progress of the function
    :return: data with embeddings
    """
    # creating a local copy of the data
    data = data.copy()

    if printouts:
        print('--- Creating simple embeddings for the data ---')

    def stanford_embedding(sen):
        # if the word is not found in the w2v model, return a zero vector of the same size
        if agg == 'sum':
            return np.sum([w2v_stanford[word.lower()] if word.lower() in w2v_stanford
                           else np.zeros(w2v_local.vector_size) for word in sen.split()], axis=0)
        elif agg == 'mean':
            return np.mean([w2v_stanford[word.lower()] if word.lower() in w2v_stanford
                           else np.zeros(w2v_local.vector_size) for word in sen.split()], axis=0)
        else:
            raise ValueError('Invalid aggregation method. Please use either "sum" or "mean"')

    def local_embedding(sen):
        if agg == 'sum':
            return np.sum([w2v_stanford[word.lower()] if word.lower() in w2v_stanford
                           else np.zeros(w2v_local.vector_size) for word in sen.split()], axis=0)
        elif agg == 'mean':
            return np.mean([w2v_stanford[word.lower()] if word.lower() in w2v_stanford
                           else np.zeros(w2v_local.vector_size) for word in sen.split()], axis=0)
        else:
            raise ValueError('Invalid aggregation method. Please use either "sum" or "mean')

    # creating the embeddings for the data
    data['stanford_embedding'] = data['title'].apply(stanford_embedding)
    data['local_embedding'] = data['title'].apply(local_embedding)
    if agg == 'sum':
        data['simple_embedding'] = data['stanford_embedding'] + data['local_embedding']
    elif agg == 'mean':
        data['simple_embedding'] = (data['stanford_embedding'] + data['local_embedding']) / 2
    else:
        raise ValueError('Invalid aggregation method. Please use either "sum" or "mean"')
    if printouts:
        print('--- Simple embeddings created ---\n')
    return data

def stacked_embedding(data:pd.DataFrame, w2v_stanford: KeyedVectors, w2v_local: Word2Vec,
                     agg: str = 'sum', printouts: bool = True) -> pd.DataFrame:
    """
    Used to create stacked embeddings for the sentences in the data.
    :param data: data to create embeddings for
    :param w2v_stanford: stanford word2vec model
    :param w2v_local: local word2vec model
    :param agg: aggregation method to use for the embeddings. Either 'sum' or 'mean', default is 'sum'
    :param printouts: whether to print out the progress of the function
    :return: data with embeddings
    """
    # creating a copy of the data
    data = data.copy()

    if printouts:
        print('--- Creating Stacked embedding for the Data ---')

    data['title_len'] = data['title'].apply(lambda x: len(x.split()))
    std_len = data['title_len'].std()
    max_len = data['title_len'].max()
    emb_stack_size = np.ceil(max_len + 2*std_len).astype(np.int64)  # for an approximate 95% coverage

    def create_embedding(sen):
        emb_stack = np.zeros((emb_stack_size, w2v_local.vector_size))
        for i, word in enumerate(sen.split()):
            if i >= emb_stack_size:
                break
            word_emb = np.zeros(w2v_local.vector_size)
            if word.lower() in w2v_stanford:
                word_emb += w2v_stanford[word.lower()]
            if word.lower() in w2v_local.wv:
                word_emb += w2v_local.wv[word.lower()]
            if agg == 'mean':
                word_emb /= 2
            emb_stack[i] = word_emb
        return emb_stack

    data['stacked_embedding'] = data['title'].apply(create_embedding)

    if printouts:
        print('--- Stacked embeddings created ---\n')
    return data

def per_word_embedding(data:pd.DataFrame, w2v_stanford: KeyedVectors, w2v_local: Word2Vec,
                     agg: str = 'sum', printouts: bool = True) -> pd.DataFrame:
    """
    Used to create per-word embeddings for the sentences in the data.
    :param data: data to create embeddings for
    :param w2v_stanford: stanford word2vec model
    :param w2v_local: local word2vec model
    :param agg: how to aggregate the embeddings, either 'sum' or 'mean'
    :param printouts: whether to print out the progress of the function
    :return: data with embeddings
    """
    # creating a copy of the data
    data = data.copy()

    if printouts:
        print('--- Creating per-word embeddings for the data ---')

    def create_embedding(sen):
        emb_list = []
        for word in sen.split():
            word_emb = np.zeros(w2v_local.vector_size)
            if word.lower() in w2v_stanford:
                word_emb += w2v_stanford[word.lower()]
            if word.lower() in w2v_local.wv:
                word_emb += w2v_local.wv[word.lower()]
            if agg == 'mean':
                word_emb /= 2
            emb_list.append(word_emb)
        return emb_list

    data['per_word_embedding'] = data['title'].apply(create_embedding)
    if printouts:
        print('--- Per-word embeddings created ---\n')

    return data