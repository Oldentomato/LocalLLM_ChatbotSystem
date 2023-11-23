from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import torch
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCauslaLM, AutoTokenizer
# from .peft2hf import HuggingFaceHugs

from langchain import PromptTemplate, LLMChain

model_path = "C:\\Users\\유아이네트웍스\\Desktop\\projects\\chatbot_python\\src\\data\\checkpoint-662/"
adapter_path = "./data/checkpoint-662"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

def read_pdfs(pdf_files):
    pages = []
    temp_filepath = os.path.join("./data", pdf_files)
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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text="", update_interval=2):
        self.text = initial_text
        self.token_buffer = []
        self.update_interval = update_interval  # 토큰이 몇 개 모일 때 UI를 업데이트할지 결정

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_buffer.append(token)
        
        # 버퍼에 저장된 토큰의 수가 update_interval에 도달하면 UI를 업데이트
        if len(self.token_buffer) >= self.update_interval:
            self.text += ''.join(self.token_buffer)
            print(self.token_buffer)
            self.token_buffer = []

#peft에서 huggingface로 변환하는 작업
m = AutoModelForCauslaLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
m = PeftModelForCausalLM.from_pretrained(m, adapter_path)
m = m.merge_and_unload()

m.save_pretrained("../out/peft_hf") #여기서 모델만 잘 저장된다면 아래들은 문제없음


# hf_model = HuggingFaceHugs(model=m, tokenizer=tokenizer)

stream_handler = StreamHandler()

#for test
template = """ Hey llama, you like to eat quinoa. Whatever question I ask you, you reply with "Waffles, waffles, waffles!".
 Question: {input} Answer: """
prompt = PromptTemplate(template=template, input_variables=["input"])
chain = LLMChain(prompt=prompt, llm=hf_model)

chain("Who is Princess Momo?")
#################################################

# embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
# pages = read_pdfs("pdf_files.pdf")

# # Split
# texts = split_pages(pages)

# persist_directory = None  # "./data_store"
# db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
#     documents=texts,
#     embedding=embeddings,
#     persist_directory=persist_directory)

# # Set retriever and LLM
# retriever = db.as_retriever()


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True)

# response = qa_chain("user_input")