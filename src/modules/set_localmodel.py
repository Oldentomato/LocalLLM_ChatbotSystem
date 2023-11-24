from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import torch
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .custom.textstreamer import TextStreamer
import tempfile


class Set_LocalModel:
    def __init__(self):
        self.model = "/prj/src/data/7b-hf"



    def get_llm_model(self):
        
        print("model load")
        pre_model = AutoModelForCausalLM.from_pretrained(
        self.model,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(self.model)

        streamer = TextStreamer(tokenizer, skip_prompt=True)

        pipe = pipeline(
            "text-generation", model=pre_model, tokenizer=tokenizer, max_new_tokens=200, streamer=streamer
        )
        hf_model = HuggingFacePipeline(pipeline=pipe)

    def get_embedding_model(self):


        print("embedding model load")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(cache_folder = "/prj/src/cache", model_name=self.model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        self.embeddings.client.tokenizer.pad_token = self.embeddings.client.tokenizer.eos_token


    def pdf_embedding(self, pdf_files):
        def read_pdfs(pdf_files):
            with tempfile.TemporaryDirectory() as temp_dir:
                pages = []
                for file in pdf_files:
                    temp_filepath = os.path.join(temp_dir, file.name)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_filepath)
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


        pages = read_pdfs(pdf_files)
        texts = split_pages(pages)

        persist_directory = "/prj/src/data_store" 
        print("pdf embedding")
        db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
            documents=texts,
            embedding=self.embeddings,
            persist_directory=persist_directory)

        print("pdf embedd and saved")




    def run_QA(self, question, mode, model):
        db = Chroma(persist_directory="/prj/src/data_store" , embedding_function=self.embeddings)
        db.get() 

        # Set retriever and LLM
        retriever = db.as_retriever(search_kwargs={"k": 2})
        print("qa_chain load")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.hf_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True)



        qa_chain(question)

    