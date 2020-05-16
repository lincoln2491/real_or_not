import os
import pickle

import numpy as np
import bcolz


class GloveMapper:
    __UNKNOWN_TOKEN__ = '<unk>'
    PAD_TOKEN = '<pad>'

    # __PAD_ID__ = -1

    def __init__(self, glove_root_path, dim):
        self.__adjusted__ = False
        self.__embedding_dim__ = dim
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

        self.__glove__ = {w: vectors[word_2_idx[w]] for w in words}
        self.__glove_word_2_idx__ = word_2_idx

    def adjust(self, all_possible_words):
        # TODO how to handle unknown  words
        self.__mapper_word_to_idx__ = {}
        weights_matrix = []
        curr_id = 0
        for word in all_possible_words:
            try:
                weights_matrix.append(self.__glove__[word])
                self.__mapper_word_to_idx__[word] = curr_id
                curr_id += 1
            except KeyError:
                pass
        weights_matrix.append(self.__glove__[self.__UNKNOWN_TOKEN__])
        self.__mapper_word_to_idx__[self.__UNKNOWN_TOKEN__] = curr_id
        curr_id += 1
        weights_matrix.append([0] * self.__embedding_dim__)
        self.__mapper_word_to_idx__[self.PAD_TOKEN] = curr_id
        self.__adjusted__ = True
        return np.array(weights_matrix), self.__mapper_word_to_idx__[self.PAD_TOKEN]

    def convert_data_set(self, sentences, max_size):
        # TODO create own error
        if not self.__adjusted__:
            raise ValueError('mapper not adjusted')
        unknown_id = self.__mapper_word_to_idx__[self.__UNKNOWN_TOKEN__]
        converted_data_set = []
        for sentence in sentences:
            new_sentence = [self.__mapper_word_to_idx__.get(word, unknown_id) for word in sentence]
            new_sentence = new_sentence + [self.__mapper_word_to_idx__[self.PAD_TOKEN]] * (max_size - len(new_sentence))
            converted_data_set.append(new_sentence)

        return converted_data_set
