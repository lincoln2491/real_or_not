from collections import Counter

import pandas as pd

from real_or_not.embeddings_mappers.glove_mapper import GloveMapper
from real_or_not.utils import clear_text

train_df = pd.read_csv('data/train.csv', keep_default_na=False)
train_df.text = train_df.text.apply(clear_text)
train_df.text = train_df.text.str.split()
gl = GloveMapper('data/glove.6B', 50)

all_words = train_df.text.to_list()
all_worlds_flat = [item for sublist in all_words for item in sublist]
tmp = [gl._glove.__contains__(k) for k in all_worlds_flat]
print(Counter(tmp))
all_worlds_flat = list(set(all_worlds_flat))
tmp = [gl._glove.__contains__(k) for k in all_worlds_flat]
print(Counter(tmp))
tmp = [k for k in all_worlds_flat if not gl._glove.__contains__(k)]
