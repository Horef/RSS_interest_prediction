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


def load_stanford_w2v() -> KeyedVectors:
    """
    Used to load the pre-trained glove model from the Stanford website.
    :return: pre-trained glove model
    """
    # loading the glove model from the glove.840B.300d.txt file, if it exists
    # if not, download the glove model
    print('--- Loading the Stanford w2v model ---')

    try:
        print('Trying to load the Stanford w2v model')
        w2v_model = KeyedVectors.load(os.path.join(os.path.dirname(__file__), STANFORD_W2V_PATH))
    except FileNotFoundError:
        print('w2v model not found, downloading stanford glove model')

        url = 'https://nlp.stanford.edu/data/glove.840B.300d.zip'
        filename = 'glove.840B.300d.zip'

        # if the file is already downloaded, skip the download
        if not os.path.exists(os.path.join(os.path.dirname(__file__), filename)):
            try:
                request.urlretrieve(url, filename)
            except urllib.error.URLError:
                print('Main link is down, trying the mirror link')
                try:
                    mirror = 'https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'
                    request.urlretrieve(mirror, filename)
                except urllib.error.URLError:
                    print('Both links are down, trying a third link')
                    try:
                        mirror = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'
                        request.urlretrieve(mirror, filename)
                    except urllib.error.URLError:
                        raise ConnectionError('All links are down, please try again later')
        else:
            print('Glove model already downloaded. Skipping the download step')

        print('Glove model downloaded, extracting the files')
        # if the file is already extracted, skip the extraction
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt')):
            with ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(__file__))
        else:
            print('Glove model already extracted. Skipping the extraction step')

        print('Converting the glove model to word2vec format')
        glove_file = datapath(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt'))
        word2vec_glove_file = get_tmpfile("glove.840B.300d.word2vec.txt")
        num_vec, dim_vec = glove2word2vec(glove_file, word2vec_glove_file)
        w2v_model = KeyedVectors.load_word2vec_format(word2vec_glove_file, limit=num_vec - 1)
        w2v_model.save(STANFORD_W2V_PATH)

        print('Cleaning up the files')
        os.remove(os.path.join(os.path.dirname(__file__), filename))
        os.remove(os.path.join(os.path.dirname(__file__), 'glove.840B.300d.txt'))

    print('--- Stanford w2v model loaded ---\n')
    return w2v_model


def train_w2v_model(sentences: [[str]], vector_size: int = 300, window: int = 2, min_count: int = 1, workers: int = 12,
                    epochs: int = 100) -> Word2Vec:
    """
    Used to train a word2vec model on a given corpus.
    :param sentences: corpus to train the model on.
    :param vector_size: desired size of the word vectors.
    :param window: window size.
    :param min_count: minimum count of a word to be included.
    :param workers: number of workers to use.
    :param epochs: number of epochs.
    :return: trained word2vec model.
    """
    print('--- Training (or updating) local word2vec model ---')
    # if there is a pre-trained model, load it and update it
    if os.path.exists(LOCAL_W2V_PATH):
        print('Loading the pre-trained word2vec model')
        w2v_model = Word2Vec.load(LOCAL_W2V_PATH)
        w2v_model.build_vocab(sentences, update=True)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epochs)
        w2v_model.save(LOCAL_W2V_PATH)
        print('--- Word2Vec model updated and saved ---\n')
        return w2v_model
    else:
        print('No pre-trained word2vec model found')
        print('Training a new word2vec model')
        # Initializing the word2vec model
        w2v_model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                             min_count=min_count, workers=workers, epochs=epochs,
                             seed=SEED)
        w2v_model.save(LOCAL_W2V_PATH)
        print('--- Word2Vec model trained and saved ---\n')
        return w2v_model


def simple_embedding(data: pd.DataFrame, w2v_stanford: KeyedVectors, w2v_local: Word2Vec,
                     agg: str = 'sum') -> pd.DataFrame:
    """
    Used to create simple embeddings for the sentences in the data.
    Each sentence is encoded as a single vector, which is the aggregation of the word vectors in the sentence.
    :param data: data to create embeddings for
    :param w2v_stanford: pre-trained word2vec model from Stanford
    :param w2v_local: pre-trained word2vec model from the local data
    :param agg: aggregation method to use for the embeddings. Either 'sum' or 'mean', default is 'sum'
    :return: data with embeddings
    """
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
    print('--- Simple embeddings created ---\n')
    return data
