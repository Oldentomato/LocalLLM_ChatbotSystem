from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
from .config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import pickle
from soylemma import Lemmatizer
from konlpy.tag import Okt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os.path   

class Embedding_Document:

    def __init__(self, save_vector_dir):
        self.save_vector_dir = save_vector_dir
    
    def get_embedding_model(self):
        print("embedding model load")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.embeddings.client.tokenizer.pad_token = self.embeddings.client.tokenizer.eos_token
        self.embeddings.client.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __read_pdfs(self, pdf_files):
        if not pdf_files:
            return []
        
        pages = []
        for path in pdf_files:
            loader = PyPDFLoader(path)
            for page in loader.load_and_split():
                pages.append(page)

        return pages


    def __split_pages(self, pages):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 0,
            length_function = len,
            is_separator_regex = False,
        )
        texts = text_splitter.split_documents(pages)
        return texts



    def bert_embedding(self, pdf_files):
        try:
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)

            persist_directory = "/prj/src/data_store" 
            print("pdf embedding")
            db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
                documents=texts,
                embedding=self.embeddings,
                persist_directory=persist_directory)

            print("pdf embedd and saved")
            print("document summaring")
            print(f"총 분할된 도큐먼트 수: {len(texts)}")
            

            # Map 단계에서 처리할 프롬프트 정의
            # 분할된 문서에 적용할 프롬프트 내용을 기입합니다.
            # 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입되니다.
            map_template = """다음은 문서 중 일부 내용입니다.
            {pages}
            이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
            답변:"""
            map_prompt = PromptTemplate.from_template(map_template)
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
            map_chain = LLMChain(llm=llm, prompt=map_prompt)

            # Reduce 단계에서 처리할 프롬프트 정의
            reduce_template = """다음은 요약의 집합입니다.:
            {doc_summaries}
            이것들을 바탕으로 통합된 요약을 만들어 주세요.
            답변:"""
            reduce_prompt = PromptTemplate.from_template(reduce_template)

            reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

            # 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain,
                document_variable_name="doc_summaries"
            )

            # Map 문서를 통합하고 순차적으로 Reduce합니다.
            reduce_documents_chain = ReduceDocumentsChain(
                # 호출되는 최종 체인입니다.
                combine_documents_chain=combine_documents_chain,
                # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
                collapse_documents_chain=combine_documents_chain,
                # 문서를 그룹화할 때의 토큰 최대 개수입니다.
                token_max=4000,
            )

            # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
            map_reduce_chain = MapReduceDocumentsChain(
                # Map 체인
                llm_chain=map_chain,
                # Reduce 체인
                reduce_documents_chain=reduce_documents_chain,
                # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
                document_variable_name="pages",
                # 출력에서 매핑 단계의 결과를 반환합니다.
                return_intermediate_steps=False,
            )

            self.summary = map_reduce_chain.run(texts)
            print(f"pdf문서 요약: {self.summary}")

            return True, None
        except Exception as e:
            print("error to embedding")
            print(f"error_msg: {e}")
            return False, e
        

    def __find_elements_with_specific_value(self, tuple_list, target_value):
        result_list = [t[0] for t in tuple_list if t[1] == target_value]
        return result_list
    
    def __find_highest_doc_index(self, result_list):
        max_value = float('-inf') #초기 최댓값을 음의 무한대로 설정
        max_index = None

        for i, sublist in enumerate(result_list):
            if len(sublist) > 0 and isinstance(sublist[0], (int, float)):
                value = sublist[0]
                if value > max_value:
                    max_value = value
                    max_index = i

        return max_index

    def __sentence_tokenizing(self, query):
        lemmatizer = Lemmatizer()
        t = Okt()
        lemm_sentence = ''
        stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        query = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", query)
        for text in t.pos(query):
            if text[0] in stopwords:
                continue
            result_lemm = self.__find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
            if len(result_lemm) == 0:
                lemm_sentence += f"{text[0]} "
            else:
                # print(result_lemm)
                lemm_sentence += f"{result_lemm[0]} "

        return lemm_sentence
    
    def embedding_tf_idf(self, pdf_files):
        try:
            print("read_pdfs")
            pages = self.__read_pdfs(pdf_files)
            print("split_pdfs")
            texts = self.__split_pages(pages)

            content = []
            source = []

            if os.path.isfile(f'{self.save_vector_dir}/content.pkl'):
                with open(f'{self.save_vector_dir}/content.pkl', 'rb') as f:
                    content = pickle.load(f)

            print(content)
            origin_content = []
            for text in texts:
                origin_content.append(text.page_content)
                result_sentence = self.__sentence_tokenizing(text.page_content)
                content.append(result_sentence)
                source.append((text.metadata['source'],text.metadata['page']))


            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(content)
            doc_info = {
                "content": content,
                "origin_content": origin_content,
                "source": source
            }
            
            # # save
            with open(f'{self.save_vector_dir}/content.pkl', 'wb') as f:
                pickle.dump(doc_info, f)
            with open(f'{self.save_vector_dir}/test2.pkl', 'wb') as f:
                pickle.dump(tfidf_matrix, f)
            with open(f'{self.save_vector_dir}/tfidf_vectorizer.pkl', 'wb') as file:
                pickle.dump(vectorizer, file)
        except Exception as e:
            print(f"error:{e}")
            return False, e
        
        return True, None
        



    def tf_idf_search_doc(self, query, k):
        with open(f'{self.save_vector_dir}/test2.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(f'{self.save_vector_dir}/tfidf_vectorizer.pkl', 'rb') as file:
            loaded_vectorizer = pickle.load(file)
        with open(f'{self.save_vector_dir}/content.pkl', 'rb') as file:
            doc_info = pickle.load(file)


        origin_content = doc_info["origin_content"]
        # content = doc_info["content"]
        source = doc_info["source"]

        new_query = self.__sentence_tokenizing(query)
        query_vector = loaded_vectorizer.transform([new_query])

        similarity_scores = cosine_similarity(tfidf_matrix, query_vector)
        result_index = self.__find_highest_doc_index(similarity_scores)

        print(origin_content[result_index])
        print(source[result_index])
        print(similarity_scores[result_index])

        return origin_content[result_index], source[result_index][0], source[result_index][1], similarity_scores[result_index][0]

