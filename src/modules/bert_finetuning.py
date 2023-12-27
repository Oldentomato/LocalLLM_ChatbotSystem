from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import re
from datasets import load_dataset

def train_bert(model, dataset_name):
    BATCH_SIZE = 20

    dataset = load_dataset(dataset_name)
    num_train = 2000
    num_validation = 500

    stand_id = "6548850-0"

    train_dataset = []
    val_corpus = {}
    val_queries = {}
    val_relevant_docs = {}
    relevant_arr = []

    for con, que in zip(dataset["train"]["context"][:num_train], dataset["train"]["question"][:num_train]):
        train_dataset.append(InputExample(texts=[que, con]))

    for id, con, que in zip(dataset["validation"]["id"][:num_validation], dataset["validation"]["context"][:num_validation], dataset["validation"]["question"][:num_validation]):
        temp_id = id.split('-')
        new_id = temp_id[0] + temp_id[1]
        if new_id == stand_id:
            relevant_arr.append(id)
        else:
            val_relevant_docs[stand_id] = relevant_arr
            stand_id = new_id
            relevant_arr = []
            relevant_arr.append(id)

        val_corpus[new_id] = con
        val_queries[new_id] = que


        

    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE
    )

    loss = losses.MultipleNegativesRankingLoss(model)


    evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant_docs)

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

#debugging
if __name__ == "__main__":
    embeddings = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
    train_bert(embeddings,"squad_kor_v1")