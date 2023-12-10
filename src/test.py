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

lemmatizer = Lemmatizer()


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
    t = Okt()
    lemm_sentence = ''
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    query = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", query)
    for text in t.pos(query):
        if text[0] in stopwords:
            continue
        result_lemm = find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
        if len(result_lemm) == 0:
            lemm_sentence += f"{text[0]} "
        else:
            # print(result_lemm)
            lemm_sentence += f"{result_lemm[0]} "

    return lemm_sentence


def find_highest_doc_index(result_list):
    max_value = float('-inf') #초기 최댓값을 음의 무한대로 설정
    max_index = None

    for i, sublist in enumerate(result_list):
        if len(sublist) > 0 and isinstance(sublist[0], (int, float)):
            value = sublist[0]
            if value > max_value:
                max_value = value
                max_index = i

    return max_index

print("read_pdfs")
pages = read_pdfs(["/prj/upload/corona.pdf"])
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

persist_directory = "/prj/src/tf_data_store"

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(content)


# # save
with open(f'{persist_directory}/test2.pickle', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
with open(f'{persist_directory}/tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# load
with open(f'{persist_directory}/test2.pickle', 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open(f'{persist_directory}/tfidf_vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

# feature_names = vectorizer.get_feature_names_out() #시각화
# df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)#시각화

# print(df_tfidf) #pdf 문장들 vectorizing 결과

query = "코로나바이러스-19란 무엇인가요?"
new_query = sentence_tokenizing(query)
query_vector = loaded_vectorizer.transform([new_query])

# query_feature_names = vectorizer.get_feature_names_out()#시각화
# df_tfidf_query = pd.DataFrame(query_vector.toarray(), columns=query_feature_names)#시각화



#유사성 계산
similarity_scores = cosine_similarity(tfidf_matrix, query_vector)
result_index = find_highest_doc_index(similarity_scores)
print(f"내용: {origin_content[result_index]}\n")
print(f"소스: {source[result_index]}\n")
print(f"점수: {similarity_scores[result_index]}")
print(f"전체점수: {similarity_scores}")


