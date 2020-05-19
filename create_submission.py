import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from real_or_not.models.SimpleModel import SimpleModel
from real_or_not.testing import predict_on_model
from real_or_not.training import train_model
from real_or_not.utils import get_datasets, preprocess_dataset, get_dataloader, get_mapper

# PARAMETERS
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 4
OWN_EMBEDDINGS = False
EMBEDDINGS_DIMENSION = 50
HIDDEN_DIMENSION = 10
NUM_LSTM_LAYERS = 1
BIDIRECTIONAL = False

# SETUP
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# READ AND PREPROCESS DATA
train_df, predict_df = get_datasets()
train_df = preprocess_dataset(train_df)
predict_df = preprocess_dataset(predict_df)
max_size = train_df.text.apply(len).max()

# MAP WORDS TO EMBEDDINGS
all_possible_words = list(set([item for sublist in train_df.text.to_list() for item in sublist]))

mapper = get_mapper(OWN_EMBEDDINGS, EMBEDDINGS_DIMENSION, all_possible_words)
vocab_size = mapper.get_pad_id() + 1

train_x = mapper.convert_data_set(train_df.text.to_list(), max_size)
train_y = train_df.target.tolist()
train_loader = get_dataloader(train_x, train_y, BATCH_SIZE, True)

predict_x = mapper.convert_data_set(predict_df.text, max_size)
predict_loader = get_dataloader(predict_x, None, BATCH_SIZE, False)

# SETUP TRAINING
net = SimpleModel(vocab_size, max_size, EMBEDDINGS_DIMENSION, HIDDEN_DIMENSION, NUM_LSTM_LAYERS, BIDIRECTIONAL)
if not OWN_EMBEDDINGS:
    net.initialize_weights(mapper.weights_matrix, mapper.get_pad_id())

net.to(device)
loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

# TRAIN
train_model(net, train_loader, None, EPOCHS, loss_function, optimizer, device)

# CREATE SUBMISSION FILE
_, predict_predicted = predict_on_model(net, predict_loader, device)

predict_df = predict_df.assign(target=predict_predicted)
predict_df[['id', 'target']].to_csv('data/sub.csv', index=None, sep=',')
