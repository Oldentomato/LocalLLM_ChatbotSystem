from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, pipeline
# from .peft2hf import HuggingFaceHugs
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


model_path = "/prj/src/data/7b-hf"
adapter_path = "/prj/src/out/model/llama-7-int4-dolly/checkpoint-1986/"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}

def read_pdfs(pdf_files):
    pages = []
    temp_filepath = os.path.join("./data/", pdf_files)
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

class TextStreamer:
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(f">>{text}", flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""

        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


#"/prj/src/out/peft_hf"
#"beomi/llama-2-ko-7b"
#"/prj/src/data/7b-hf"
model = "/prj/src/data/7b-hf"
#peft에서 huggingface로 변환하는 작업
# m = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map={"": 0}
# )

# m = PeftModelForCausalLM.from_pretrained(m, adapter_path)
# m = m.merge_and_unload()
# tok = LlamaTokenizerFast.from_pretrained(model)
# tok.bos_token_id = 1
# tok.pad_token_id = 0
# m.config.pad_token_id = tok.pad_token_id # unk
# m.config.bos_token_id = 1
# m.config.eos_token_id = 2
# m.eval()
# m.save_pretrained("./out/test") #여기서 모델만 잘 저장된다면 아래들은 문제없음

print("model load")
pre_model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)
tokenizer = LlamaTokenizerFast.from_pretrained(model)

streamer = TextStreamer(tokenizer, skip_prompt=True)

pipe = pipeline(
    "text-generation", model=pre_model, tokenizer=tokenizer, max_new_tokens=200, streamer=streamer
)
hf_model = HuggingFacePipeline(pipeline=pipe)

# template="""질문: {question}

# 답변: """

# prompt = PromptTemplate.from_template(template)
# llm_chain = LLMChain(prompt=prompt, llm=hf_model)
# print(llm_chain.run(question="한국의 수도는 어디야?"))



print("embedding model load")
embeddings = HuggingFaceEmbeddings(cache_folder = "/prj/src/cache", model_name=model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token




pages = read_pdfs("SAMPLE.pdf")

# if tok.pad_token is None:
#     tok.add_special_tokens({'pad_token': '[PAD]'})
# embeddings.resize_token_embeddings(len(tok))

# Split
texts = split_pages(pages)

persist_directory = "./data_store"  # "./data_store"
print("pdf embedding")
db = Chroma.from_documents( #chromadb 임베딩된 텍스트 데이터들을 효율적으로 저장하기위한 모듈
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory)

# Set retriever and LLM
retriever = db.as_retriever()

print("qa_chain load")
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

print("done")
response = qa_chain("딥러닝에 대해 알려줘")

print(response)
for source in response['source_documents']:
    file_name = os.path.basename(source.metadata['source'])
    page_number = source.metadata['page']
    print(f"파일: {file_name}\n")
    print(f"페이지: {page_number}\n")