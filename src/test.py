from langchain.vectorstores import Chroma
from config import OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from soylemma import Lemmatizer
from konlpy.tag import Okt

lemmatizer = Lemmatizer()

def find_elements_with_specific_value(tuple_list, target_value):
    try:
        result_list = [t[0] for t in tuple_list if t[1] == target_value]
    except:
        return ["None"]
    else:
        return result_list
    

t = Okt()
lemm_sentence = ''
query = """창 밖의 동물들은 돼지에서 인간으로, 인간에서 돼지로, 
다시 돼지에서 인간으로 번갈아 시선을 옮겼다.
그러나 누가 돼지고 누가 인간인지, 어느 것이 어느 것인지 
이미 분간할 수 없었다."""
print(t.pos(query))
for text in t.pos(query):
    result_lemm = find_elements_with_specific_value(lemmatizer.lemmatize(text[0]),text[1]) #0 = 텍스트, 1 = 품사
    lemm_sentence += f"{result_lemm[0]},"
else:
    lemm_sentence += result_lemm[0]

print(lemm_sentence)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# def wordLemmatizer(data):
#     tag_map = defaultdict(lambda : wn.NOUN)
#     tag_map['J'] = wn.ADJ
#     tag_map['V'] = wn.VERB
#     tag_map['R'] = wn.ADV
#     file_clean_k =pd.DataFrame()
#     for index,entry in enumerate(data):
        
#         # Declaring Empty List to store the words that follow the rules for this step
#         Final_words = []
#         # Initializing WordNetLemmatizer()
#         word_Lemmatized = WordNetLemmatizer()
#         # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#         for word, tag in pos_tag(entry):
#             # Below condition is to check for Stop words and consider only alphabets
#             if len(word)>1 and word not in stopwords.words('english') and word.isalpha():
#                 word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#                 Final_words.append(word_Final)
#             # The final processed set of words for each iteration will be stored in 'text_final'
#                 file_clean_k.loc[index,'Keyword_final'] = str(Final_words)
#                 file_clean_k=file_clean_k.replace(to_replace ="\[.", value = '', regex = True)
#                 file_clean_k=file_clean_k.replace(to_replace ="'", value = '', regex = True)
#     return file_clean_k

# def read_pdfs(pdf_files):
#     if not pdf_files:
#         return []
    
#     pages = []
#     for path in pdf_files:
#         loader = PyPDFLoader(path)
#         for page in loader.load_and_split():
#             pages.append(page)

#     return pages


# def split_pages(pages):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 500,
#         chunk_overlap  = 0,
#         length_function = len,
#         is_separator_regex = False,
#     )
#     texts = text_splitter.split_documents(pages)
#     return texts



# pdf_dirs = ["../upload/corona.pdf"]
# pages = read_pdfs(pdf_dirs)
# texts = split_pages(pages)
# stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


# tokenized=[]
# for content in texts:
#     sentence = content.page_content
#     sentence = sentence.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
#     temp = tokenizer.morphs(sentence)
#     temp = [word for word in temp if not word in stopwords] #불용어 제거
#     tokenized.append(temp)


# print(text[0].page_content)
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# db = Chroma(persist_directory="../data_store")
# print(db._collection.get())