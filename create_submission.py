from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.glove_mapper import GloveMapper
from real_or_not.models.SimpleModel import SimpleModel
from real_or_not.training import train_model
from real_or_not.utils import get_datasets, preprocess_dataset, get_dataloader

LEARNING_RATE = 0.01
EPOCHS = 100
USE_EMPTY_WORD = True
BATCH_SIZE = 4
OWN_EMBEDDINGS = False
EMBEDDINGS_DIMENSION = 50
HIDDEN_DIMENSION = 10
NUM_LSTM_LAYERS = 1
BIDIRECTIONAL = False

torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(6)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_df, predict_df = get_datasets()
train_df = preprocess_dataset(train_df)
predict_df = preprocess_dataset(predict_df)

max_size = train_df.text.apply(len).max()
gm = GloveMapper('data/glove.6B', EMBEDDINGS_DIMENSION)
all_sentences = train_df.text.to_list()
all_possible_words = list(set([item for sublist in all_sentences for item in sublist]))
gm.adjust(all_possible_words)
weight_matrix = gm.weights_matrix
pad_id = gm.get_pad_id()
vocab_size = pad_id + 1
train_x = gm.convert_data_set(all_sentences, max_size)
train_y = train_df.target.tolist()
train_loader = get_dataloader(train_x, train_y, BATCH_SIZE, True)

predict_x = gm.convert_data_set(predict_df.text, max_size)

predict_loader = get_dataloader(predict_x, None, BATCH_SIZE, False)


net = SimpleModel(vocab_size, max_size, EMBEDDINGS_DIMENSION, HIDDEN_DIMENSION, NUM_LSTM_LAYERS, BIDIRECTIONAL)
if not OWN_EMBEDDINGS:
    net.initialize_weights(weight_matrix, pad_id)
net.to(device)
loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_model(net, train_loader, None, EPOCHS, loss_function, optimizer, device)

all_scores = []
with torch.no_grad():
    for i, sentences in enumerate(predict_loader):
        if i % 100 == 0:
            print(datetime.now(), i / len(predict_x) * 100)
        sentences = sentences.to(device)
        scores = net(sentences)
        all_scores.extend(scores.cpu())

all_scores = [scores.argmax() for scores in all_scores]
all_scores = [scores.tolist() for scores in all_scores]

predict_df = predict_df.assign(target=all_scores)
predict_df[['id', 'target']].to_csv('data/sub.csv', index=None, sep=',')
