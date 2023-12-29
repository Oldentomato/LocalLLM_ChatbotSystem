from sentence_transformers import SentenceTransformer, InputExample, losses, datasets
from sentence_transformers.evaluation import InformationRetrievalEvaluator

import re
from datasets import load_dataset

def train_bert(model, dataset_name):
    BATCH_SIZE = 32

    dataset = load_dataset(dataset_name)
    num_train = 20000
    num_validation = 5000

    stand_id = "6548850-0"

    train_dataset = []
    val_corpus = {}
    val_queries = {}
    val_relevant_docs = {}
    relevant_arr = []

    for con, que in zip(dataset["train"]["context"][:num_train], dataset["train"]["question"][:num_train]):
        train_dataset.append(InputExample(texts=[que, con]))

    # for id, con, que in zip(dataset["validation"]["id"][:num_validation], dataset["validation"]["context"][:num_validation], dataset["validation"]["question"][:num_validation]):
    #     temp_id = id.split('-')
    #     new_id = temp_id[0] + temp_id[1]
    #     if new_id == stand_id:
    #         relevant_arr.append(id)
    #     else:
    #         val_relevant_docs[stand_id] = relevant_arr
    #         stand_id = new_id
    #         relevant_arr = []
    #         relevant_arr.append(id)

    #     val_corpus[new_id] = con
    #     val_queries[new_id] = que


        

    loader = datasets.NoDuplicatesDataLoader(
        train_dataset, batch_size=BATCH_SIZE
    )

    loss = losses.MultipleNegativesRankingLoss(model)# 긍정과 부정 데이터셋 중에서 긍정의 데이터셋만 있을 경우 사용하는 함수
    # 즉, label은 필요가 없음. label이 없으므로 evaluator도 자연스럽게 필요가 없어짐
    #참고 https://acdongpgm.tistory.com/339#google_vignette


    # evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant_docs)

    EPOCHS = 10

    #0.1 이 전체 warmup의 비율임 여기를 조정해볼 것
    warmup_steps = int(len(loader) * EPOCHS * 0.2)
    """
    warmup의 필요성
    warm-up은 학습 초기에 모델의 가중치를 서서히 조정하여 안정적으로 학습을 시작할 수 있도록 돕는 기법이다.
    특히 학습률을 매우 낮은 값에서 점차적으로 증가시키는 방식으로 진행되며, 이는 학습 초기에 큰 학습률로 인해 발생할 수 있는
    가중치의 급격한 변화와 그로 인한 학습의 불안정성을 방지하는데 도움을 준다.
    """

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path='../../out/exp_finetune',
        show_progress_bar=True,
        # evaluator=evaluator, 
        # evaluation_steps=50,
    )

#debugging
if __name__ == "__main__":
    embeddings = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
    train_bert(embeddings,"squad_kor_v1")