from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import json
from torch.utils.data import DataLoader
import re

# define model
#"BAAI/bge-small-en"
model_id = "BM-K/KoSimCSE-roberta"
model = SentenceTransformer(model_id)

TRAIN_DATASET_FPATH = '../data/train_dataset.json'
VAL_DATASET_FPATH = '../data/val_dataset.json'

# We use a very small batchsize to run this toy example on a local machine. 
# This should typically be much larger. 
BATCH_SIZE = 10

with open(TRAIN_DATASET_FPATH, 'r+', encoding='utf-8') as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, 'r+', encoding='utf-8') as f:
    val_dataset = json.load(f)

dataset = train_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = InputExample(texts=[query, text])
    examples.append(example)


loader = DataLoader(
examples, batch_size=BATCH_SIZE
)

loss = losses.MultipleNegativesRankingLoss(model)


dataset = val_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)

EPOCHS = 10

warmup_steps = int(len(loader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='../../out/exp_finetune',
    show_progress_bar=True,
    evaluator=evaluator, 
    evaluation_steps=50,
)