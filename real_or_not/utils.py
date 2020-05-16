import string
import re
import pandas as pd
import unidecode


def clear_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = unidecode.unidecode(text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text
