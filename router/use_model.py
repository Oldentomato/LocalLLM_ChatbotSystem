from fastapi import APIRouter
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import threading
import queue
from pydantic import BaseModel
from fastapi import Form
from src.modules import Set_LocalModel
from typing import List

usemodel = APIRouter()
local_model = Set_LocalModel()
local_model.get_llm_model()
local_model.get_embedding_model()

class ThreadGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item
    
    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class RequestItem(BaseModel):
    query: str
    model_name: str



def chat_llama(query):
    g = ThreadGenerator()
    threading.Thread(target=local_model.run_QA, args=(g, query)).start()
    return g


@usemodel.post("/pdfembedding")
async def llamaquery(pdfs: List[UploadFile]):
    success,e = local_model.pdf_embedding(pdfs)
    return {"success":success, "error": e}

@usemodel.post("/llamaquery")
async def llamaquery(query: str = Form(...)):
    return StreamingResponse(chat_llama(query=query), media_type='text/event-stream')