# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import sqlite3
import tempfile
import streamlit as st
import modules

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# 라마, gpt3.5, gpt4 중 택 1 옵션 넣기 < 해결
# 여러 pdf문서 넣을 수 있게 하기(프레스바 넣기)
# 임베딩된 값들 저장하고 다시 불러올 수 있게 하기
# 전체 답변과
# 각 출처에서 추출하여 같이 제공하기

# 제목
st.title("SearchingPDFs")
st.write("---")


# 파일 업로드
pdf_files = st.file_uploader("PDF 파일을 올려주세요!", accept_multiple_files=True, type=['pdf'])
st.write("---")
chat = modules.Chat_UI(is_debugging=False)#채팅 모듈
local_model = modules.Set_LocalModel()
model = None

select_model = st.radio(
    "Select to Mode",
    ["Embedding_PDF", "Search"],
    index=1
)

st.write("You selected:", select_model)


def Answer_Bot():
    with st.spinner("모델 로드중"):
        model = local_model.get_llm_model()
    user_input = chat.on_input_change()
    local_model.run_QA(user_input,None,model)

# self.hf_model , self.embeddings에서 return으로 나오도록 바꾸고
# 해당 변수의 값이 할당되어있는지 확인하고 안되어있으면 모델로드하도록 변경
if select_model == "Search":

    # Question
    st.header("KnowBot!!")

    chat.display_chat()
    with st.container():
        
        st.text_input("User Input:", on_change=Answer_Bot, key="user_input")

elif select_model == "Embedding_PDF":
    if pdf_files:
        if st.button('입력'):
            with st.spinner("모델 로드중"):
                local_model.get_embedding_model()
            local_model.pdf_embedding(pdf_files)



