'''
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure
make
make install

(test)mecab -d /usr/local/lib/mecab/dic/mecab-ko-dic 
curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
pip install mecab-python
pip install nltk
pip install gensim
pip install torchtext
'''
from konlpy.tag import Mecab
import urllib.request
import pandas as pd
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')
#debug
data[:10]

#한글과 공백을 제외하고 모두 제거
data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

#불용어 정의
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

tokenizer = Mecab()
tokenized=[]
for sentence in data['document']:
    temp = tokenizer.morphs(sentence)
    temp = [word for word in temp if not word in stopwords] #불용어 제거
    tokenized.append(temp)

vocab = FreqDist(np.hstack(tokenized)) #빈도수 계산도구인 FreqDist()로 빈도수 계산

#상위 n개의 단어만 단어 집합으로 저장 
vocab_size = 500

#상위 n개의 단어만 보존
vocab = vocab.most_common(vocab_size)
print(f'단어 집합의 크기 : {len(vocab)}')
print(vocab[:10])


#인덱스 0과 1은 다른 용도로 남겨두고 나머지 단어들은 2부터 105까지 순차적으로 인덱스 부여
word_to_index = {word[0] : index+2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

#기존 훈련 데이터에서 각 단어를 고유한 정수로 부여
encoded = []
for line in tokenized:
    temp = []
    for w in line:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['unk'])
        
    encoded.append(temp)


max_len = max(len(l) for l in encoded)
print(f'문장의 최대 길이 : {max_len}')
print(f'문장의 최소 길이 : {min(len(l) for l in encoded)}')
print(f'문장의 평균 길이 : {sum(map(len, encoded))/len(encoded)}')
plt.hist([len(s) for s in encoded], bins=50)
plt.xlabel('length of sample')
plt.ylabel('number of sample')
plt.show()

#모든 문장의 길이를 63으로 통일
for line in encoded:
    if len(line) < max_len: #현재 샘플이 정해준 길이보다 짧으면
        line += [word_to_index['pad'] * (max_len - len(line))] #나머지는 전부 pad 토큰으로 채운다

print(f'문장의 최대 길이 : {max(len(l) for l in encoded)}')
print(f'문장의 최소 길이 : {min(len(l) for l in encoded)}')
print(f'문장의 평균 길이 : {sum(map(len, encoded))/len(encoded)}')


#Word2Vec으로 벡터화
model = Word2Vec(sentences = tokenized, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
'''
    size = 워드 벡터의 특징 값, 임베딩된 벡터의 차원
    window = 컨텍스트 윈도우 크기
    min_count = 단어 최소 빈도 수 제한(빈도가 적은 타이틀은 학습하지 않음)
    workers = 학습을 위한 프로세스 수
    sg = 0은 CBOW, 1은 Skip-gram
'''

#debug
model.wv.vectors.shape

model.save(".out/tokenizer/word2vec_model.model")

