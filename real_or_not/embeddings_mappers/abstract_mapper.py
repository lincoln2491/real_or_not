from abc import ABC, abstractmethod


class AbstractMapper(ABC):
    _UNKNOWN_TOKEN = '<unk>'
    _PAD_TOKEN = '<pad>'

    def __init__(self, dim):
        self._adjusted = False
        self._embedding_dim = dim

    @abstractmethod
    def adjust(self, all_possible_words):
        pass

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

    def get_pad_id(self):
        return self._mapper_word_to_idx[self._PAD_TOKEN]
