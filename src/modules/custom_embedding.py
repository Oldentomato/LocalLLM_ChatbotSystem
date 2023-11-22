import torch
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, GPTTreeIndex, PromptHelper, QuestionAnswerPrompt
from llama_index import LLMPredictor, ServiceContext, LangchainEmbedding
from llama_index.indices.tree.select_leaf_embedding_retriever import TreeSelectLeafEmbeddingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import StorageContext, load_index_from_storage
from typing import Optional, List, Mapping, Any
from chatglm import ChatGLM
from config import LLM_DEVICE
import logging, csv, sys, codecs, time



max_input_size = 2048
# set number of output tokens
num_output = 200
# set maximum chunk overlap
max_chunk_overlap = 20
embedding_path = "/path/to/text2vec-large-chinese"
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embedding_path))
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# define our LLM
# ChatGLM is a custom LLM defined using langchain interface
llm_predictor = LLMPredictor(llm=ChatGLM(model_name_or_path="/path/to/chatglm-6b",
                  llm_device=LLM_DEVICE))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model)

# Load the your data
documents = SimpleDirectoryReader('/path/to/data').load_data()

# define prompt
QA_PROMPT_TMPL = (
   "Question:"
   "{context_str}"
   "\n---------------------\n"
   "Answer:"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
index = GPTListIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()
query_engine = index.as_query_engine(retriever_mode="embedding", text_qa_template=QA_PROMPT)