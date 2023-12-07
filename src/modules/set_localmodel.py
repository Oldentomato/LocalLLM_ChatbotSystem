from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain, LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.prompts import PromptTemplate
from .custom.textstreamer import TextStreamer
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from .config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

#"beomi/llama-2-ko-7b"
#"jhgan/ko-sroberta-multitask"
#"/prj/out/exp_finetune"
#"beomi/kcbert-base"

class Set_LocalModel:
    def __init__(self):
        self.model = "beomi/llama-2-ko-7b"
        self.embedd_model = "beomi/kcbert-base"
        self.chat_history = []
        self.context = ""


    def read_summary(self):
        with open("/prj/src/data_store/summary.txt", "r") as file:
            self.summary = file.read()

        print(self.summary)


    def get_llm_model(self):
        print("model load")
        self.pre_model = AutoModelForCausalLM.from_pretrained(
        self.model,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
        )

        self.pre_model.eval()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model)



    def get_embedding_model(self):
        print("embedding model load")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedd_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.embeddings.client.tokenizer.pad_token = self.embeddings.client.tokenizer.eos_token
        self.embeddings.client.tokenizer.add_special_tokens({'pad_token': '[PAD]'})



    def pdf_embedding(self, pdf_files):
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


        try:
            print("read_pdfs")
            pages = read_pdfs(pdf_files)
            print("split_pdfs")
            texts = split_pages(pages)

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

    def __set_prompt(self):
        prompt_template = """당신은 문서검색 지원 에이전트입니다.\n
        요약 정보를 제공하면 그 정보를 활용하여 대답해주세요. 요약 정보는 아래에 있습니다.\n
        {context}
        대화 내역을 활용하여 추가적으로 답변을 생성해도 됩니다. 대화 내역은 아래와 같습니다.\n
        {chat_history}
        답을 모르면 모른다고만 하고 답을 만들려고 하지 마세요. 같은 말은 반복하지 마세요.\n
        참고된 pdf의 내용이 없다면, 없다고 답하세요.\n
        """
        system_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history"]
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)

        question_prompt = PromptTemplate(
            template="""아래 문장에 대한 답변을 말해주세요.\n
            {question}\n
            답변:""",
            input_variables=["question"]
        )

        question_message_prompt = HumanMessagePromptTemplate(prompt=question_prompt)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, question_message_prompt])


        return chat_prompt


    def run_QA(self, g, question):
        try:
            db = Chroma(persist_directory="/prj/src/data_store" , embedding_function=self.embeddings)
            db.get() 


            streamer = TextStreamer(g=g, tokenizer=self.tokenizer, skip_prompt=True)

            pipe = pipeline(
                "text-generation", 
                model=self.pre_model, 
                repetition_penalty=1.5, #높을수록 반복성을 방지시킴
                tokenizer=self.tokenizer, 
                return_full_text = False, 
                max_new_tokens=200, #max_length는 프롬프트를 포함한 최대길이를 지정, max_new_tokens는 프롬프트를 제외한 최대길이를 지정함
                streamer=streamer,
            )
            res = pipe( #__call__
                do_sample = True, # False로 해두면 Greedy Search만 진행되어 같은 결과만 생성하게 된다.
                no_repeat_ngram_size=2, #생성시간이 숫자에 비례해서 올라가지만 문장 생성시에 반복되는 경우를 줄일 수 있다.
                temperature=0.2, # 낮을수록 가장 확률이 높은 토큰을 선택하게됨. 높을수록 더 창의적인 답변이 나오게됨 (낮음: 사실적, 높음: 창의적)
                #top_k와 top_p는 greedy가 False(do_sample이 True)인 경우에 활용된다.
                top_k=0, #확률이 높은 순서대로 k번째까지 높은 단어에 대해 필터링한다. 값이 높을수록 무작위 샘플 방식에 가까워지고 낮을수록 탐욕 방식에 가까워진다.
                top_p=0.95, #여기서 top_k를 0으로두고 top_p의 값만 할당해주면 뉴클러스 샘플링이라고 하게 된다. 확률이 가장 높은 순으로 후보 단어들을 더했을 때 top_p가 되는단어 집합에 대해 샘플링한다.
                
                #즉, greedy를 True로 하면 항상 확률이 높은 단어만 선택하기때문에 편향적이고 일관된 문장만 출력하게되고, 경우에 따라 반복되는 단어가 나올수 있고,
                #greedy를 False로 하면 top_k와top_p의 샘플링 방식을 통해 좀 더 자연스럽고 다양한 문장들을 출력하게 된다.


            )
            hf_model = HuggingFacePipeline(pipeline=res)

            memory = ConversationBufferMemory(memory_key="chat_history", human_prefix="question", ai_prefix="answer", return_messages=True)

            # Set retriever and LLM
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # 3개의 답변을 생성하고 유사도를 기준으로 가장 높은 점수를 최종답변으로 도출함

            #compression_retriever
            print("qa_chain load")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=hf_model,
                chain_type="stuff", #map_rerank
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt":self.__set_prompt()}
                )


            #, "context" : self.context, "chat_history": self.chat_history
            response = qa_chain({"question":question, "chat_history": self.chat_history, "context": self.summary})
            self.chat_history= [(question, response["answer"])]

            g.send(f"\n파일: {os.path.basename(response['source_documents'][0].metadata['source'])}") #3개중에 하나고르는거다보니까 0번으로해버리면 정확해지지가 않음
            g.send(f"\n페이지: {response['source_documents'][0].metadata['page']}")


        except Exception as e:
            print('Failed:'+str(e))
        finally:
            g.close()


    