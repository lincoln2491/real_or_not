import string
import re
import pandas as pd
import unidecode
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.embeddings_mappers.glove_mapper import GloveMapper
from real_or_not.embeddings_mappers.own_embeddings_mapper import OwnEmbeddingsMapper
from real_or_not.embeddings_mappers.twitter_mapper import TwitterMapper


def clear_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', 'url', text)
    text = re.sub(r'@[a-zA-Z0-9_]*', 'username', text)
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


def get_mapper(own_embeddings, embeddings_type, dim, all_possible_words):
    if own_embeddings:
        mapper = OwnEmbeddingsMapper(dim)
    else:
        mapper = GloveMapper('data/glove.6B', dim) if embeddings_type == 'GLOVE' else TwitterMapper(
            'data/glove.twitter.27B', dim)
    mapper.adjust(all_possible_words)
    return mapper


def get_dataloader(x, y, batch_size, is_train):
    train_ds = TextDataset(x, y)
    train_loader = DataLoader(train_ds, batch_size, shuffle=is_train, num_workers=2)
    return train_loader
