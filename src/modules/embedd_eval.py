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
    dataset = load_dataset("squad_kor_v1")
    num_example = 3
    contexts = []
    questions = []

    for con, que in zip(dataset["validation"]["context"][:num_example], dataset["validation"]["question"][:num_example]):
        contexts.append(sentence_tokenizing(con, "string"))
        questions.append(sentence_tokenizing(que, "string"))
    
    # print(contexts)
    # print(questions)
    return contexts, questions




def embedding_doc2vec(query):
    content = []

    tagged_data = [TaggedDocument(words=content, tags=[str(id)]) for id, content in enumerate(content)]

    max_epochs = 10

    model = Doc2Vec(
        window=20, #모델 학습할 때 앞뒤로 보는 단어의 수
        vector_size=300, #벡터 차원의 크기
        alpha=0.025, #lr
        min_alpha=0.025,
        min_count=5, #학습에 사용할 최소 단어 빈도 수
        dm=0, #학습방법 1=PV-DM, 0=PV_DBOW
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

    new_query = sentence_tokenizing(query, "array")
    inferred_vector = model.infer_vector(new_query)

    return inferred_vector



def embedding_tf_idf(contexts_list, answers_list):
    vectorizer = TfidfVectorizer()
    contexts_tfidf = vectorizer.fit_transform(contexts_list).toarray()
    answers_tfidf = vectorizer.transform(answers_list).toarray()


    # Perform t-SNE for visualization
    contexts_tsne = TSNE(n_components=2,perplexity=2, n_jobs=1).fit_transform(contexts_tfidf)
    answers_tsne = TSNE(n_components=2,perplexity=2, n_jobs=1).fit_transform(answers_tfidf)

    return contexts_tsne, answers_tsne


def Embedding_Compare(predict_emb_list, true_emb_list):
    embeddings_tsne = [TSNE(n_components=2).fit_transform(embedding) for embedding in predict_emb_list]

    plt.figure(figsize=(15,6))

    for i, (tsne_result, answer) in enumerate(zip(embeddings_tsne, true_emb_list)):
        plt.subplot(1,3, i+1)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], label=f'Embedding Model {i+1}', color=f'C{i}')
        plt.title(f'Embedding Model {i+1}')

        cos_similarity = cosine_similarity(tsne_result, answer.reshape(1,-1))
        avg_cos_similarity = np.mean(cos_similarity)

        print(f'Embedding Model {i+1}과의 평균 코사인 유사도: {avg_cos_similarity}')
    
    plt.show()


context, question = Set_Dataset()
context_tsne, answer_tsne = embedding_tf_idf(context, question)
Embedding_Compare(context_tsne, question)



