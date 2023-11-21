import os
from collections import Counter
from typing import Optional, List, Union
from konlpy.tag import Mecab
import urllib.request
import pandas as pd
from nltk import FreqDist
from tokenizers import BertWordPieceTokenizer, CharBPETokenizer
from transformers import BertTokenizer, PreTrainedTokenizerFast

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
# data = pd.read_table('ratings.txt')
# #debug
# # print(data.iloc[46471])

# #한글과 공백을 제외하고 모두 제거
# data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
# data = data.replace('', pd.NA)

# #결측치 제거
# data = data.dropna(axis=0)


# tokenizer = Mecab()
# with open("../out/prepare.mecab.txt", "w", encoding="utf-8") as f:
#     for sentence in data['document']:
#         f.write(f'{tokenizer.morphs(sentence)}\n')
    

vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5


special_tokens = ["<s>", "</s>", "<unk>","</s>"]


tokenizer = CharBPETokenizer(lowercase=False, unicode_normalizer='nfc')
# tokenizer = BertWordPieceTokenizer()

tokenizer.train(
    files="../out/prepare.mecab.txt",
    vocab_size=vocab_size,
    # limit_alphabet=limit_alphabet,
    special_tokens = special_tokens,
    # min_frequency=min_frequency
)

tokenizer.save('../out/bertvocab/vocab.json')

#transformers형태로 저장
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../out/bertvocab/vocab.json")
tokenizer.save_pretrained("../out/berttoken/")

#bert transformers형태로 저장
# bert_token = BertTokenizer('../out/bertvocab/vocab.txt')
# bert_token.save_pretrained('../out/berttoken/')


