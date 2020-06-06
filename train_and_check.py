import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from real_or_not.models.SimpleModel import SimpleModel
from real_or_not.testing import predict_on_model
from real_or_not.training import train_model
from real_or_not.utils import get_datasets, preprocess_dataset, get_dataloader, get_mapper

# PARAMETERS
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 4
OWN_EMBEDDINGS = False
EMBEDDINGS_DIMENSION = 50
HIDDEN_DIMENSION = 10
NUM_LSTM_LAYERS = 1
BIDIRECTIONAL = True
USE_ALL_WORDS_FOR_EMBEDDINGS = True
FREEZE_EMBEDDINGS = True
USE_DROPOUT = False
EMBEDDINGS_TYPE = 'TWITTER'

assert EMBEDDINGS_TYPE in ['GLOVE', 'TWITTER'], 'EMBEDDINGS_TYPE should be GLOVE or TWITTER'

# SETUP
torch.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# READ AND PREPROCESS DATA
train_df, _ = get_datasets()
train_df = preprocess_dataset(train_df)
max_size = train_df.text.apply(len).max()

# SPLIT DATA
train_df, val_df = train_test_split(train_df, train_size=0.66, random_state=6)

# MAP WORDS TO EMBEDDINGS
if USE_ALL_WORDS_FOR_EMBEDDINGS:
    all_possible_words = list(
        set([item for sublist in (train_df.text.to_list() + val_df.text.to_list()) for item in sublist]))
else:
    all_possible_words = list(set([item for sublist in train_df.text.to_list() for item in sublist]))

mapper = get_mapper(OWN_EMBEDDINGS, EMBEDDINGS_TYPE, EMBEDDINGS_DIMENSION, all_possible_words)
vocab_size = mapper.get_pad_id() + 1

train_x = mapper.convert_data_set(train_df.text.to_list(), max_size)
train_y = train_df.target.tolist()
train_loader = get_dataloader(train_x, train_y, BATCH_SIZE, True)

val_x = mapper.convert_data_set(val_df.text.to_list(), max_size)
val_y = val_df.target.tolist()
val_loader = get_dataloader(val_x, val_y, BATCH_SIZE, False)

# SETUP TRAINING
net = SimpleModel(vocab_size, max_size, EMBEDDINGS_DIMENSION, HIDDEN_DIMENSION, NUM_LSTM_LAYERS, BIDIRECTIONAL,
                  USE_DROPOUT)
if not OWN_EMBEDDINGS:
    net.initialize_weights(mapper.weights_matrix, mapper.get_pad_id(), FREEZE_EMBEDDINGS)

net.to(device)
loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

# TRAIN
train_model(net, train_loader, val_loader, EPOCHS, loss_function, optimizer, device)

# VALIDATE
train_actual, train_predicted = predict_on_model(net, train_loader, device)
val_actual, val_predicted = predict_on_model(net, val_loader, device)

train_f1_score = f1_score(train_actual, train_predicted)
val_f1_score = f1_score(val_actual, val_predicted)
print(f'final_train_f1: {train_f1_score}')
print(f'final_val_f1: {val_f1_score}')
