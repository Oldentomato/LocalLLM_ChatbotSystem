import os
from collections import Counter
from typing import Optional, List, Union
from konlpy.tag import Mecab
import urllib.request
import pandas as pd
from nltk import FreqDist
from tokenizers import BertWordPieceTokenizer


class KoNLPyPreTokenizer:
    def __init__(self, konlpy_tagger):
        self.konlpy_tagger = konlpy_tagger

    def __call__(self, sentence):
        return self.pre_tokenize(sentence)

    def pre_tokenize(self, sentence):
        return ' '.join(self.konlpy_tagger.morphs(sentence))

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')
#debug
# print(data.iloc[46471])

#한글과 공백을 제외하고 모두 제거
data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
data = data.replace('', pd.NA)

#결측치 제거
data = data.dropna(axis=0)


tokenizer = Mecab()
with open("./out/prepare.mecab.txt", "w", encoding="utf-8") as f:
    for sentence in data['document']:
        f.write(f'{tokenizer.morphs(sentence)}\n')
    

vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5


tokenizer = BertWordPieceTokenizer(lowercase=False, trip_accents=False)

tokenizer.train(
    files="./out/prepare.mecab.txt",
    vocab_size=vocab_size,
    limit_alphabet=limit_alphabet,
    min_frequency=min_frequency
)

tokenizer.save_model(directory='./out/', name="bert_tokenizer")