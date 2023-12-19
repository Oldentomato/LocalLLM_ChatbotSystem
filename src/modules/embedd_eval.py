import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from soylemma import Lemmatizer
from konlpy.tag import Okt
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import OrderedDict

#jhgan/ko-sroberta-multitask
#beomi/kcbert-base
#BM-K/KoSimCSE-roberta-multitask
def Get_Embedding():
    embeddings = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
    return embeddings

def find_elements_with_specific_value(tuple_list, target_value):
    result_list = [t[0] for t in tuple_list if t[1] == target_value]
    return result_list

def sentence_tokenizing(query, mode="string"):
    lemmatizer = Lemmatizer()

    t = Okt()
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
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


def Set_Dataset():
    #squad_kor_v1 = Train: 60407 Validation: 5774
    dataset = load_dataset("squad_kor_v1")
    num_example = 5000
    contexts = []
    questions = []

    for con, que in zip(dataset["validation"]["context"][:num_example], dataset["validation"]["question"][:num_example]):
        contexts.append(con)
        questions.append(que)
    
    # print(contexts)
    # print(questions)
    return contexts, questions



def mrr_measure(predict_list):
    score = 0
    for predict in predict_list:
        if 1 not in predict:
            continue
        score += 1 / (predict.index(1) + 1)

    return score / len(predict_list)

def top_N_precision(predict_list, k):
    c, m = [0] * k, 0
    for idx, predict in enumerate(predict_list):
        if 1 in predict:
            c[predict.index(1)] += 1

        m += 1
    top_n_precision = [sum(c[:idx + 1]) / m for idx, e in enumerate(c)]

    return top_n_precision
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def reranking(embeddings, query, top_documents, doc_score, k):
    alpha = 0.9
    '''
    alpha:
    가중치 조절: alpha 값은 코사인 유사도에 대한 가중치를 조절하는 매개변수입니다. 
    이 가중치를 통해 사용자는 초기 문서의 중요도와 코사인 유사도의 중요도를 조절할 수 있습니다. 
    만약 alpha가 1에 가까우면 코사인 유사도가 더 중요해지고, 0에 가까우면 초기 문서의 점수가 더 중요해집니다.
    '''
    query_vector = embeddings.encode([query])
    corpus_embeddings = embeddings.encode(top_documents)
    similarity_scores = cosine_similarity(query_vector, corpus_embeddings)

    cos_scores = similarity_scores[0]
    cos_scores = softmax(cos_scores)
    result_scores = (1-alpha)*np.array(doc_score) + alpha*cos_scores 
    #result_scores는 초기 문서의 정보와 코사인 유사도를 적절히 조합하여 최종 결과를 얻기 위해 사용됩니다.

    top_results = np.argpartition(-result_scores, range(k))[0:k]
    rerank_documents = [top_documents[i] for i in top_results]
    rerank_scores = [result_scores[i] for i in top_results]

    return rerank_scores, rerank_documents


def bertSearch(embeddings, query, contexts, embedd_contexts, k):
    query_vector = embeddings.encode([query])
    similarity_scores = cosine_similarity(query_vector, embedd_contexts)
    cos_scores = similarity_scores[0]
    cos_scores = softmax(cos_scores)
    document_idx = np.argpartition(-cos_scores, range(k))[0:k]

    top_documents = [contexts[i] for i in document_idx]

    return top_documents


def Search(bm25, contexts, query, k):
    arr_new_query = sentence_tokenizing(query, "array")
    doc_scores = bm25.get_scores(arr_new_query)
    doc_scores = softmax(doc_scores)
    document_idx = np.argpartition(-doc_scores, range(k))[0:k]
    top_documents = [contexts[i] for i in document_idx]
    doc_scores = [doc_scores[i] for i in document_idx]

    return top_documents, doc_scores



def evaluate(k):
    #top_documents는 토크나이징 안한 평문임
    bmbert_predict_lists = []
    bm_predict_lists = []
    bert_predict_lists = []
    embedding = Get_Embedding()
    
    contexts, questions = Set_Dataset()
    token_context = []
    embedding_context = []
    set_context = list(OrderedDict.fromkeys(contexts))
    for con in tqdm(set_context):
        arr_new_query = sentence_tokenizing(con, "array")
        token_context.append(arr_new_query)
        embedding_context.append(embedding.encode(con))

    bm25 = BM25Okapi(token_context)


    for q,a in tqdm(zip(questions, contexts)):
        bm_top_documents, doc_score = Search(bm25, set_context, q, k)
        q = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", q)
        _, top_documents = reranking(embedding, q, bm_top_documents, doc_score, k)

        bert_top_documents = bertSearch(embedding, q, set_context, embedding_context, k)
        
        #bmbert
        bool_documents = [0]*k
        for idx, document in enumerate(top_documents):
            bool_documents[idx] = int(a == document)

        bmbert_predict_lists.append(bool_documents)

        #bm25
        bool_documents = [0]*k
        for idx, document in enumerate(bm_top_documents):
            bool_documents[idx] = int(a == document)

        bm_predict_lists.append(bool_documents)

        #bert
        bool_documents = [0]*k
        for idx, document in enumerate(bert_top_documents):
            bool_documents[idx] = int(a == document)

        bert_predict_lists.append(bool_documents)

        

    print('BMBERT MRR Score : ', mrr_measure(bmbert_predict_lists))
    print('BMBERT Top10 Precision Score : ',top_N_precision(bmbert_predict_lists, k))
    print('BM25 MRR Score : ', mrr_measure(bm_predict_lists))
    print('BM25 Top10 Precision Score : ',top_N_precision(bm_predict_lists, k))
    print('BERT MRR Score : ', mrr_measure(bert_predict_lists))
    print('BERT Top10 Precision Score : ',top_N_precision(bert_predict_lists, k))


if __name__ == "__main__":
    evaluate(k=10)




