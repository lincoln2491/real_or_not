import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from real_or_not.TextDataset import TextDataset
from real_or_not.models.SimpleModel import SimpleModel
from sklearn.metrics import f1_score

BATCH_SIZE = 4

pd.set_option('display.max_columns', 20)
MAX_SIZE = 33
EMPTY_WORD = '<EMPTY>'


def extend_y(y):
    return [0] * y[0] + [1] * (y[1] - y[0] + 1) + [0] * (MAX_SIZE - y[1] - 1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_df = pd.read_csv('data/train.csv', keep_default_na=False)
predict_df = pd.read_csv('data/test.csv', keep_default_na=False)

train_df.text = train_df.text.str.split()
predict_df.text = predict_df.text.str.split()

all_words = train_df.text.to_list()
all_words = [t + [EMPTY_WORD] * (MAX_SIZE - len(t)) for t in all_words]
all_worlds_flat = [item for sublist in all_words for item in sublist]

le = LabelEncoder()
words_labeled = le.fit_transform(all_worlds_flat)

train_x = []

# for i, sentence in enumerate(all_words):
#     if i % 1000:
#         print(datetime.now(), i / len(all_words) * 100)
#     labeled_sentence = le.transform(sentence)
#     train_x.append(labeled_sentence)
with open('data/train.pickle', 'rb') as f:
    train_x = pickle.load(f)
train_y = train_df.target.tolist()

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.66, random_state=6)

all_words_predict = predict_df.text.to_list()
all_words_predict = [t + [EMPTY_WORD] * (MAX_SIZE - len(t)) for t in all_words_predict]

vocab_size = le.classes_.__len__() + 1

# predict_x = []
# for i, sentence in enumerate(all_words_predict):
#     if i % 1000:
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
with open('data/test.pickle', 'rb') as f:
    predict_x = pickle.load(f)

train_ds = TextDataset(train_x, train_y)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
val_ds = TextDataset(val_x, val_y)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)

predict_ds = TextDataset(predict_x)
predict_loader = DataLoader(predict_ds)

net = SimpleModel(vocab_size, MAX_SIZE, 30, 20)
net.to(device)

loss_function = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
for epoch in range(1000):
    print(f'{datetime.now()}: epoch {epoch}')
    for i, (sentences, targets) in enumerate(train_loader):
        if i % 1000 == 0:
            print(f'{datetime.now()}: {((i * BATCH_SIZE) / len(train_ds)) * 100}')
        sentences = sentences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        tag_scores = net(sentences)

        loss = loss_function(tag_scores, targets)
        # print(loss)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_predicted = []
        val_predicted = []
        train_actual = []
        val_actual = []
        for sentences, targets in train_loader:
            sentences = sentences.to(device)
            tag_scores = net(sentences)
            train_predicted.extend(tag_scores.cpu().numpy())
            train_actual.extend(targets.numpy())
        for sentences, targets in val_loader:
            sentences = sentences.to(device)
            tag_scores = net(sentences)
            val_predicted.extend(tag_scores.cpu().numpy())
            val_actual.extend(targets.numpy())
        train_predicted = np.array(train_predicted)
        val_predicted = np.array(val_predicted)
        train_predicted = train_predicted.argmax(axis=1)
        val_predicted = val_predicted.argmax(axis=1)
        print(
            f'epoch: {epoch}: train loss {f1_score(train_actual, train_predicted)} val loss {f1_score(val_actual, val_predicted)}')

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
