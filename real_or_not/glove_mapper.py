import os
import pickle

import numpy as np
import bcolz


class GloveMapper:
    _UNKNOWN_TOKEN = '<unk>'
    _PAD_TOKEN = '<pad>'

    # __PAD_ID__ = -1

    def __init__(self, glove_root_path, dim):
        self._adjusted = False
        self._embedding_dim = dim
        glove_cache_path = os.path.join(glove_root_path, f'6B.{dim}.dat')
        glove_embeddings_path = os.path.join(glove_root_path, f'glove.6B.{dim}d.txt')
        glove_words_path = os.path.join(glove_root_path, f'6b.{dim}_words.pkl')
        glove_words_2_idx_path = os.path.join(glove_root_path, f'6b.{dim}_idx.pkl')
        if not os.path.isdir(glove_cache_path) or \
                not os.path.isfile(glove_words_path) or \
                not os.path.isfile(glove_words_2_idx_path):
            words = []
            idx = 0
            word_2_idx = {}
            vectors = bcolz.carray(np.zeros(1), rootdir=glove_cache_path, mode='w')
            with open(glove_embeddings_path, 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word = line[0]
                    words.append(word)
                    word_2_idx[word] = idx
                    idx += 1
                    vect = np.array(line[1:]).astype(np.float)
                    vectors.append(vect)
            vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=glove_cache_path, mode='w')
            vectors.flush()
            pickle.dump(words, open(glove_words_path, 'wb'))
            pickle.dump(word_2_idx, open(glove_words_2_idx_path, 'wb'))
        else:
            vectors = bcolz.open(glove_cache_path)[:]
            words = pickle.load(open(glove_words_path, 'rb'))
            word_2_idx = pickle.load(open(glove_words_2_idx_path, 'rb'))

        self._glove = {w: vectors[word_2_idx[w]] for w in words}
        self._glove_word_2_idx = word_2_idx

    def adjust(self, all_possible_words):
        # TODO how to handle unknown  words
        self._mapper_word_to_idx = {}
        weights_matrix = []
        curr_id = 0
        for word in all_possible_words:
            try:
                weights_matrix.append(self._glove[word])
                self._mapper_word_to_idx[word] = curr_id
                curr_id += 1
            except KeyError:
                pass
        weights_matrix.append(self._glove[self._UNKNOWN_TOKEN])
        self._mapper_word_to_idx[self._UNKNOWN_TOKEN] = curr_id
        curr_id += 1
        weights_matrix.append([0] * self._embedding_dim)
        self._mapper_word_to_idx[self._PAD_TOKEN] = curr_id
        self._adjusted = True
        self.weights_matrix = np.array(weights_matrix)

    def get_pad_id(self):
        return self._mapper_word_to_idx[self._PAD_TOKEN]

    def convert_data_set(self, sentences, max_size):
        # TODO create own error
        if not self._adjusted:
            raise ValueError('mapper not adjusted')
        unknown_id = self._mapper_word_to_idx[self._UNKNOWN_TOKEN]
        converted_data_set = []
        for sentence in sentences:
            new_sentence = [self._mapper_word_to_idx.get(word, unknown_id) for word in sentence]
            new_sentence = new_sentence + [self._mapper_word_to_idx[self._PAD_TOKEN]] * (
                    max_size - len(new_sentence))
            converted_data_set.append(new_sentence)

        return converted_data_set
