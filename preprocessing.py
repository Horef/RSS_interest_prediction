import torch

from constants import BACKUP_PATH, SEED
import pandas as pd
import os
import pickle
from torch.utils.data import Dataset


def load_train_data(data_path: str, tagged: bool = True) -> pd.DataFrame:
    """
    Load the training data from the given data_path
    :param data_path: path to the data
    :param tagged: whether the data is tagged or not
    :return: pandas DataFrame of the data
    """
    print('--- Loading training data ---')

    # appending lines in the data_path to a full backup file
    with open(BACKUP_PATH, 'a') as backup_file:
        with open(data_path) as f:
            for line in f:
                backup_file.write(line)
    print('Data backed up to backup.txt')

    # if some data has already been loaded, load it and append the new data
    if os.path.exists('train_data.pkl'):
        print('train_data.pkl exists, appending the new data')
        data = pd.read_pickle('train_data.pkl')
        if tagged:
            new_data = pd.read_csv(data_path, header=None, names=['title', 'interest'])
        else:
            new_data = pd.read_csv(data_path, header=None, names=['title'])
        data = pd.concat([data, new_data])
        data.to_pickle('train_data.pkl')

        # cleaning the data from the data_path
        with open(data_path, 'w') as f:
            pass
        print('Data appended to train_data.pkl')
        print('--- Training data loaded ---\n')
        return data

    else:
        print('train_data.pkl does not exist, creating it')
        if tagged:
            data = pd.read_csv(data_path, header=None, names=['title', 'interest'])
        else:
            data = pd.read_csv(data_path, header=None, names=['title'])
        data.to_pickle('train_data.pkl')

        # cleaning the data from the data_path
        with open(data_path, 'w') as f:
            pass
        print('Data loaded to train_data.pkl')
        print('--- Training data loaded ---\n')
        return data


def read_file(file_path: str, tagged: bool = True) -> pd.DataFrame:
    """
    Read a file and return a pandas DataFrame
    :param file_path: path to the file
    :param tagged: whether the file is tagged or not
    :return: dataframe of the file
    """
    if tagged:
        return pd.read_csv(file_path, header=None, names=['title', 'interest'])
    return pd.read_csv(file_path, header=None, names=['title'])


def clean_data(data: pd.DataFrame, printouts: bool = True) -> pd.DataFrame:
    """
    Clean the data by removing any rows with missing values, and adding spaces around every special character
    :param data: data to clean
    :param printouts: whether to print out the cleaning steps
    :return: cleaned data
    """
    if printouts:
        print('--- Cleaning data ---')
    data.dropna(inplace=True)

    # adding spaces around special characters
    data['title'] = data['title'].str.replace(r'([^\w\s])', r' \1 ', regex=True)

    # trimming consecutive spaces
    data['title'] = data['title'].str.replace(r' +', ' ', regex=True)

    # removing duplicate rows if any
    data.drop_duplicates(subset=['title'], keep='last', inplace=True)

    # setting all words to lowercase
    data['title'] = data['title'].str.lower()

    if printouts:
        print('Data cleaned')
        print('--- Data cleaned ---\n')
    return data


def split_data(data: pd.DataFrame, train_percent: float = 0.8, shuffle: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """
    Split the data into train and test such that the representation of the interest column is the same in both
    :param data: data to split
    :param train_percent: percentage of data to be used for training (0 < train_percent < 1)
    :param shuffle: whether to shuffle the data after splitting
    :return: train and test data
    """
    # Because we want to keep the same representation of the interest column in both train and test
    # we need to split the data into positive and negative data and then split each separately
    positive_data = data[data['interest'] == 1]
    negative_data = data[data['interest'] == 0]

    # Splitting the data
    positive_train = positive_data.sample(frac=train_percent, random_state=SEED)
    positive_test = positive_data.drop(positive_train.index)
    negative_train = negative_data.sample(frac=train_percent, random_state=SEED)
    negative_test = negative_data.drop(negative_train.index)

    # Concatenating the data
    train_data = pd.concat([positive_train, negative_train])
    test_data = pd.concat([positive_test, negative_test])

    if shuffle:
        train_data = train_data.sample(frac=1, random_state=SEED).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_data, test_data


def fold_split(data: pd.DataFrame, folds: int = 5) -> [pd.DataFrame]:
    """
    Split the data into folds
    :param data: data to split
    :param folds: number of folds
    :return: list of dataframes
    """
    # creating a copy of the data
    data = data.copy()

    # Because we want to keep the same representation of the interest column in both train and test
    # we need to split the data into positive and negative data and then split each separately
    positive_data = data[data['interest'] == 1]
    negative_data = data[data['interest'] == 0]

    percent_per_fold = 1 / folds
    folded_data = []
    for i in range(folds):
        positive_fold = positive_data.sample(frac=percent_per_fold, random_state=SEED)
        positive_data = positive_data.drop(positive_fold.index)
        negative_fold = negative_data.sample(frac=percent_per_fold, random_state=SEED)
        negative_data = negative_data.drop(negative_fold.index)
        folded_data.append(pd.concat([positive_fold, negative_fold]))

    return folded_data

class TextDataset(Dataset):
    def __init__(self, data: pd.DataFrame, emb_column: str):
        self.data = data
        self.emb_column = emb_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data.iloc[idx][self.emb_column], dtype=torch.float32),
                torch.tensor(self.data.iloc[idx]['interest'], dtype=torch.float32))