import nest_asyncio

nest_asyncio.apply()
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import OpenAI
from IPython.display import Markdown, display
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import QueryBundle
import pandas as pd
from IPython.display import display, HTML

OPENAI_API_KEY = "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"

llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model="local")

documents = SimpleDirectoryReader("./data/retrieve").load_data()

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)



def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
            service_context=service_context,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    print(result_dicts)
    # pretty_print(pd.DataFrame(result_dicts))


while(1):
    query = input("query: ")
    if query =="end":
        break
    
    new_nodes = get_retrieved_nodes(
        query,
        vector_top_k=3,
        with_reranker=False,
    )

    visualize_retrieved_nodes(new_nodes)

# query_engine = index.as_query_engine(
#     similarity_top_k=10,
#     node_postprocessors=[reranker],
#     response_mode="tree_summarize",
# )
# response = query_engine.query(
#     "What did the author do during his time at Y Combinator?",
# )