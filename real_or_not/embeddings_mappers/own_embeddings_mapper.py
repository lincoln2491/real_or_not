from sklearn.preprocessing import LabelEncoder

from real_or_not.embeddings_mappers.abstract_mapper import AbstractMapper


class OwnEmbeddingsMapper(AbstractMapper):
    def __init__(self, dim):
        super().__init__(dim)

    def adjust(self, all_possible_words):
        self._mapper_word_to_idx = {}
        curr_id = 0
        for word in all_possible_words:
            try:
                self._mapper_word_to_idx[word] = curr_id
                curr_id += 1
            except KeyError:
                pass
        self._mapper_word_to_idx[self._UNKNOWN_TOKEN] = curr_id
        curr_id += 1
        self._mapper_word_to_idx[self._PAD_TOKEN] = curr_id
        self._adjusted = True
