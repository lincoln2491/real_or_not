import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):

    def __init__(self, vocabulary_size, sentence_length, embedding_dimension, hidden_dimmension, num_layers,
                 bidirectional):
        super(SimpleModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.sentence_length = sentence_length
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimmension

        self.word_embeddings = nn.Embedding(self.vocabulary_size + 1, self.embedding_dimension,
                                            padding_idx=-1)
        self.lstm = nn.LSTM(self.embedding_dimension, self.hidden_dimension, batch_first=True, num_layers=num_layers,
                            bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.hidden_dimension * self.sentence_length, 2)

    def initialize_weights(self, weights, pad_id):
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(weights, dtype=torch.float32), freeze=True,
                                                            padding_idx=pad_id)

    def forward(self, x):
        # print('f-sentence', x.size())
        x = self.word_embeddings(x)
        # print('f-emb', x.size())
        # x = x.view(self.sentence_length, -1, self.embedding_dimension)
        # print('f-view', x.size())
        x, _ = self.lstm(x)
        # print('f-lstm', x.size())
        x = x.reshape(-1, self.hidden_dimension * self.sentence_length)
        # print('f-view', x.size())
        x = self.fc1(x)
        # print('f-fc1', x.size())
        # x = F.log_softmax(x, dim=1)
        # print('f-soft_max', x.size())
        return x
