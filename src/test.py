import chromadb
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
from sentence_transformers import SentenceTransformer


def find_elements_with_specific_value(tuple_list, target_value):
    result_list = [t[0] for t in tuple_list if t[1] == target_value]
    return result_list

def sentence_tokenizing(query, mode="string"):
    lemmatizer = Lemmatizer()

    t = Okt()
    # stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    stopwords=[]
    query = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", query)

    if mode == "string":
        lemm_sentence = ''
        for text in t.pos(query):
            if text[0] in stopwords:
                continue
            result_lemm = find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
            if len(result_lemm) == 0:
                lemm_sentence += f"{text[0]} "
            else:
                # print(result_lemm)
                lemm_sentence += f"{result_lemm[0]} "
    elif mode == "array":
        lemm_sentence = []
        for text in t.pos(query):
            if text[0] in stopwords:
                continue
            result_lemm = find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
            if len(result_lemm) == 0:
                lemm_sentence.append(text[0])
            else:
                lemm_sentence.append(result_lemm[0])

    return lemm_sentence

# while(1):
#     query = input("query: ")
#     if query == "stop":
#         break

#     result = sentence_tokenizing(query, "array")

#     print(result)

chroma_client = chromadb.EphemeralClient()

embeddings = SentenceTransformer("beomi/kcbert-base")
# embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
# embeddings.client.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

query = "안녕하세요 만나서 반갑습니다."
vectors = embeddings.encode(sentence_tokenizing(query, "array"))
print(vectors)
# test_collection = chroma_client.create_collection(name='test', embedding_function=embedding_function)
