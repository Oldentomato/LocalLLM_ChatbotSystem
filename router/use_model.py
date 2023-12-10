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
# local_model.get_llm_model()
# local_model.get_embedding_model()
# local_model.read_summary()

#이 변수들은 product할때는 sql에서 가져오는 것으로 해야함


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
async def embedding(pdfs: List[UploadFile], mode: str = "tf-idf"):
    files = []
    for pdf in pdfs:
        with open("./upload/"+pdf.filename, "wb") as f:
            f.write(pdf.file.read())
        files.append("./upload/"+pdf.filename)
    success,e = local_model.pdf_embedding(files, mode)
    return {"success":success, "error": e}


@usemodel.post("/searchdoc")
async def search_doc(query: str = Form(...), doc_count: int = 1, mode: str = "tf-idf"):
    content, source, page, score = local_model.search_doc(query, doc_count, mode)
    return {"doc": content, "score": score, "source": source, "page": page}


@usemodel.post("/llamaquery")
async def llamaquery(query: str = Form(...)):
    return StreamingResponse(chat_llama(query=query), media_type='text/event-stream')