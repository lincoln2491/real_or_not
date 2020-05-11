import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.models.SimpleModel import SimpleModel

# parameters
from real_or_not.testing import predict_on_model
from real_or_not.training import train_model

LEARNING_RATE = 0.01
EPOCHS = 1
USE_EMPTY_WORD = True
BATCH_SIZE = 4
OWN_EMBEDDINGS = True
EMBEDDINGS_DIMENSION = 20
HIDDEN_DIMENSION = 10
print(Path().absolute())
# setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read and transform data

if OWN_EMBEDDINGS:
    empty_word = '<EMPTY>'
    train_df = pd.read_csv('data/train.csv', keep_default_na=False)
    train_df.text = train_df.text.str.split()
    max_size = train_df.text.apply(len).max()
    predict_df = pd.read_csv('data/test.csv', keep_default_na=False)
    le = LabelEncoder()

    all_words = train_df.text.to_list()
    all_words = [t + [empty_word] * (max_size - len(t)) for t in all_words]
    all_worlds_flat = [item for sublist in all_words for item in sublist]

    le = LabelEncoder()
    words_labeled = le.fit_transform(all_worlds_flat)

    # train_x = []
    # for i, sentence in enumerate(all_words):
    #     if i % 100 == 0:
    #         print(datetime.now(), i / len(all_words) * 100)
    #     labeled_sentence = le.transform(sentence)
    #     train_x.append(labeled_sentence)
    # with open('data/train.pickle', 'wb') as f:
    #     pickle.dump(train_x, f)
    with open('data/train.pickle', 'rb') as f:
        train_x = pickle.load(f)
    train_y = train_df.target.tolist()

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.66, random_state=6)
    vocab_size = le.classes_.__len__() + 1
    train_ds = TextDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    val_ds = TextDataset(val_x, val_y)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)

# train model
net = SimpleModel(vocab_size, max_size, EMBEDDINGS_DIMENSION, HIDDEN_DIMENSION)
net.to(device)
loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_model(net, train_loader, val_loader, EPOCHS, loss_function, optimizer, BATCH_SIZE, device, train_ds)

# validate

train_actual, train_predicted = predict_on_model(net, train_loader, device)
val_actual, val_predicted = predict_on_model(net, val_loader, device)

train_f1_score = f1_score(train_actual, train_predicted)
val_f1_score = f1_score(val_actual, val_predicted)
print(f'train_f1: {train_f1_score}')
print(f'val_f1: {val_f1_score}')
