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

def calculate_average_precision(precision, recall):
    """
    정밀도-재현율 곡선을 기반으로 평균 정밀도(mAP)를 계산합니다.
    """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # 정밀도-재현율 곡선을 각각의 구간에서 계산
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 구간 별 면적 계산
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def calculate_map(ground_truth, predictions):
    """
    평균 정밀도(mAP)를 계산합니다.
    """
    num_ground_truth = len(ground_truth)
    num_predictions = len(predictions)

    true_positives = np.zeros(num_predictions)
    false_positives = np.zeros(num_predictions)
    precision = np.zeros(num_predictions)

    for i, prediction_set in enumerate(predictions):
        # 각 ground_truth에 대해 true_positives와 false_positives를 계산
        for j, prediction in enumerate(prediction_set):
            if prediction['document'] == ground_truth[i]:
                true_positives[j] = 1
            else:
                false_positives[j] = 1
            precision[j] = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))

    recall = np.sum(true_positives) / num_ground_truth

    # 평균 정밀도 계산
    ap = calculate_average_precision(precision, recall)

    return ap

#deprecated
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
    
def reranking(embeddings, query, top_documents, doc_score, k):
    query_vector = embeddings.encode([query])
    corpus_embeddings = embeddings.encode(top_documents)
    similarity_scores = cosine_similarity(query_vector, corpus_embeddings)

    result_scores = similarity_scores[0]
    top_results = np.argpartition(-result_scores, range(k))[0:k]
    rerank_documents = [top_documents[i] for i in top_results]
    rerank_scores = [result_scores[i] for i in top_results]

    return rerank_scores, rerank_documents


def Search(bm25, contexts, query, k):
    arr_new_query = sentence_tokenizing(query, "array")
    doc_scores = bm25.get_scores(arr_new_query)
    document_idx = np.argpartition(-doc_scores, range(k))[0:k]
    top_documents = [contexts[i] for i in document_idx]
    doc_scores = [doc_scores[i] for i in document_idx]

    return top_documents, doc_scores


def evaluate(k):
    #top_documents는 토크나이징 안한 평문임
    bmbert_predict_lists = []
    bm_predict_lists = []
    embedding = Get_Embedding()
    
    contexts, questions = Set_Dataset()
    token_context = []
    for con in tqdm(contexts):
        arr_new_query = sentence_tokenizing(con, "array")
        token_context.append(arr_new_query)

    bm25 = BM25Okapi(token_context)
    ground_truth = []
    predictions = []
    bm_predictions = []

    for q,a in tqdm(zip(questions, contexts)):
        bm_top_documents, bm_doc_score = Search(bm25, contexts, q, k)
        doc_score, top_documents = reranking(embedding, q, bm_top_documents, bm_doc_score, k)

        ground_truth.append(a)
        predictions.append([{'document': doc, 'score': score} for doc,score in zip(top_documents, doc_score)])
        bm_predictions.append([{'document': doc, 'score': score} for doc,score in zip(bm_top_documents, bm_doc_score)])


    # mAP 계산
    mAP = calculate_map(ground_truth, predictions)
    bm25_mAP = calculate_map(ground_truth, bm_predictions)


    
    print('BMBERT mAP Score : ', mAP)
    print('BM25 mAP Score : ', bm25_mAP)



if __name__ == "__main__":
    evaluate(k=10)




