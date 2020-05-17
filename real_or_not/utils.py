import string
import re
import pandas as pd
import unidecode
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.glove_mapper import GloveMapper
from real_or_not.models.SimpleModel import SimpleModel


def clear_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = unidecode.unidecode(text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text


def get_datasets():
    train_df = pd.read_csv('data/train.csv', keep_default_na=False)
    predict_df = pd.read_csv('data/test.csv', keep_default_na=False)
    return train_df, predict_df


def preprocess_dataset(dataset):
    dataset.text = dataset.text.apply(clear_text)
    dataset.text = dataset.text.str.split()
    return dataset


def create_glove_mapper(dataset, glove_path, embedding_dim):
    gm = GloveMapper(glove_path, embedding_dim)
    all_sentences = dataset.text.to_list()
    all_possible_words = list(set([item for sublist in all_sentences for item in sublist]))
    gm.adjust(all_possible_words)
    return gm


def get_dataloader(x, y, batch_size, is_train):
    train_ds = TextDataset(x, y)
    train_loader = DataLoader(train_ds, batch_size, shuffle=is_train, num_workers=2)
    return train_loader
