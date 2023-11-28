from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, pipeline, AutoModel, AutoTokenizer
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from .custom.textstreamer import TextStreamer
import tempfile
from langchain.memory import ConversationBufferMemory


#"beomi/llama-2-ko-7b"
#"/prj/src/data/7b-hf"
class Set_LocalModel:
    def __init__(self):
        self.model = "local:./out/eval"



    def get_llm_model(self):
        
        print("model load")
        self.pre_model = AutoModelForCausalLM.from_pretrained(
        self.model,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
        )

        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model)



    def get_embedding_model(self):
        print("embedding model load")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(cache_folder = "/prj/src/cache", model_name=self.model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.embeddings.client.tokenizer.pad_token = self.embeddings.client.tokenizer.eos_token


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
                chunk_size = 300,
                chunk_overlap  = 20,
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
            return True, None
        except Exception as e:
            print("error to embedding")
            print(f"error_msg: {e}")
            return False, e


    def __set_prompt(self):
        prompt_template = """You are a Chat support agent.
        Use the following user related the chat history (delimited by <hs></hs>) to answer the question at the end:
        You should be friendly, but not overly chatty. Context information is below. Given the context information and not prior knowledge, answer the query.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Below are the chats of history of the user:\n 
        {context}
        <hs>
        {chat_history}
        </hs>
        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["chat_history", "question", "context"]
        )

        return PROMPT


    def run_QA(self, g, question):
        try:
            db = Chroma(persist_directory="/prj/src/data_store" , embedding_function=self.embeddings)
            db.get() 

            streamer = TextStreamer(g=g, tokenizer=self.tokenizer, skip_prompt=True)

            pipe = pipeline(
                "text-generation", model=self.pre_model, tokenizer=self.tokenizer, max_new_tokens=800, streamer=streamer
            )
            hf_model = HuggingFacePipeline(pipeline=pipe)

            memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)

            # Set retriever and LLM
            retriever = db.as_retriever(search_kwargs={"k": 2})
            print("qa_chain load")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=hf_model,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={"prompt":self.__set_prompt()}
                )


            qa_chain({"question":question})


        except Exception as e:
            print('Failed:'+str(e))
        finally:
            g.close()


    