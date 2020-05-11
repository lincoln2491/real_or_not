import pickle
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.models.SimpleModel import SimpleModel
from real_or_not.training import train_model

LEARNING_RATE = 0.01
EPOCHS = 1
USE_EMPTY_WORD = True
BATCH_SIZE = 4
OWN_EMBEDDINGS = True
EMBEDDINGS_DIMENSION = 20
HIDDEN_DIMENSION = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

with open('data/train.pickle', 'rb') as f:
    train_x = pickle.load(f)
train_y = train_df.target.tolist()

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.66, random_state=6)
vocab_size = le.classes_.__len__() + 1
train_ds = TextDataset(train_x, train_y)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
val_ds = TextDataset(val_x, val_y)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)

net = SimpleModel(vocab_size, max_size, EMBEDDINGS_DIMENSION, HIDDEN_DIMENSION)
net.to(device)
loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_model(net, train_loader, val_loader, EPOCHS, loss_function, optimizer, BATCH_SIZE, device, train_ds)
empty_word = '<EMPTY>'
predict_df.text = predict_df.text.str.split()
all_words_predict = predict_df.text.to_list()
all_words_predict = [t + [empty_word] * (max_size - len(t)) for t in all_words_predict]

vocab_size = le.classes_.__len__() + 1

predict_x = []
# for i, sentence in enumerate(all_words_predict):
#     if i % 1000 == 0:
#         print(datetime.now(), i / len(all_words_predict) * 100)
#     try:
#         labeled_sentence = le.transform(sentence)
#     except:
#         labeled_sentence = []
#         for word in sentence:
#             try:
#                 labeled_word = le.transform([word])[0]
#             except:
#                 labeled_word = vocab_size - 1
#             labeled_sentence.append(labeled_word)
#     predict_x.append(labeled_sentence)
#     with open('data/test.pickle', 'wb') as f:
#         pickle.dump(predict_x, f)

with open('data/test.pickle', 'rb') as f:
    predict_x = pickle.load(f)
predict_ds = TextDataset(predict_x)
predict_loader = DataLoader(predict_ds)
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
