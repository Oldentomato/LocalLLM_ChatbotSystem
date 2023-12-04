from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
import os
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from .custom.textstreamer import TextStreamer
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
                chunk_size = 200,
                chunk_overlap  = 10,
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
        prompt_template = """당신은 문서검색 지원 에이전트입니다.
        채팅 기록과 관련된 다음 사용자(<hs></hs>로 구분)를 사용하여 마지막에 질문에 답합니다.
        친근하게 대답해도 되지만 지나치게 수다를 떨면 안 됩니다. 상황 정보는 아래에 있습니다.(<cx></cx>로 구분) 사전 지식이 아닌 상황 정보가 주어지면 질문에 답하십시오.
        답을 모르면 모른다고만 하고 답을 만들려고 하지 마세요. 같은 말은 반복하지 마세요.
        채팅 기록은 다시 답변에서 말하지 마세요.\n 
        <cx>{context}</cx>
        <hs>{chat_history}</hs>
        """

        sys_prompt: PromptTemplate = PromptTemplate(
            input_variables=["chat_history", "context"],
            template=prompt_template
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)
        

        question_prompt: PromptTemplate = PromptTemplate(
            input_variables=["question"],
            template=
            """질문에 대한 답변을 말해주세요.
            질문: {question}
            답변: """
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
                "text-generation", model=self.pre_model, repetition_penalty=1.1, tokenizer=self.tokenizer, return_full_text = True, max_new_tokens=200, streamer=streamer
            )
            hf_model = HuggingFacePipeline(pipeline=pipe)

            memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)

            # Set retriever and LLM
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})


            #compression_retriever
            print("qa_chain load")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=hf_model,
                chain_type="stuff", #map_rerank
                retriever=retriever,
                return_source_documents=True,
                memory=memory,
                combine_docs_chain_kwargs={"prompt":self.__set_prompt()}
                )

            print(self.context)
            print(self.chat_history)
            #, "context" : self.context, "chat_history": self.chat_history
            response = qa_chain({"question":question})
            self.chat_history.append((question, response["answer"]))
            self.context = response["answer"]
            # print(response)

            g.send(f"\n파일: {os.path.basename(response['source_documents'][0].metadata['source'])}")
            g.send(f"\n페이지: {response['source_documents'][0].metadata['page']}")


        except Exception as e:
            print('Failed:'+str(e))
        finally:
            g.close()


    