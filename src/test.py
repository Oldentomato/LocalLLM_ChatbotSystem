from langchain.vectorstores import Chroma
# from config import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from soylemma import Lemmatizer
from konlpy.tag import Okt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer


def read_pdfs(pdf_files):
    if not pdf_files:
        return []
    
    pages = []
    for path in pdf_files:
        loader = PyPDFLoader(path)
        for page in loader.load_and_split():
            pages.append(page)

    return pages


def split_pages(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)
    return texts


def find_elements_with_specific_value(tuple_list, target_value):
    result_list = [t[0] for t in tuple_list if t[1] == target_value]
    return result_list

def sentence_tokenizing(query):
    lemmatizer = Lemmatizer()
    t = Okt()
    lemm_sentence = []
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    query = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", query)
    for text in t.pos(query):
        if text[0] in stopwords:
            continue
        result_lemm = find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
        if len(result_lemm) == 0:
            lemm_sentence.append(text[0])
        else:
            # print(result_lemm)
            lemm_sentence.append(result_lemm[0])

    return lemm_sentence


print("read_pdfs")
pages = read_pdfs(["../upload/traffic.pdf"])
print("split_pdfs")
texts = split_pages(pages)

content = []
source = []
origin_content = []

for text in texts:
    origin_content.append(text.page_content)
    result_sentence = sentence_tokenizing(text.page_content)
    content.append(result_sentence)
    source.append((text.metadata['source'],text.metadata['page']))

tagged_data = [TaggedDocument(words=content, tags=[str(id)]) for id, content in enumerate(content)]

max_epochs = 10

model = Doc2Vec(
    window=20, #모델 학습할 때 앞뒤로 보는 단어의 수
    vector_size=300, #벡터 차원의 크기
    alpha=0.025, #lr
    min_alpha=0.025,
    min_count=5, #학습에 사용할 최소 단어 빈도 수
    dm=1, #학습방법 1=PV-DM, 0=PV_DBOW
    negative=5, #Complexity Reduction 방법, negative sampling
    seed=9999
)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print(f'iteration {epoch}')
    model.train(tagged_data, 
                total_examples=model.corpus_count,
                epochs=max_epochs)

    model.alpha -= 0.002
    model.min_alpha = model.alpha

model.random.seed(9999)

query = "양보차로 설치"
new_query = sentence_tokenizing(query)
inferred_vector = model.infer_vector(new_query)
return_docs = model.docvecs.most_similar(positive=[inferred_vector], topn=5)

for rd in return_docs:
    index = int(rd[0])
    print(f"문서내용: {origin_content[index]}\n")
    print(f"점수: {rd[1]}")

