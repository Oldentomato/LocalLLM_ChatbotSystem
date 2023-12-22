from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

# Fine-tuning을 위한 데이터셋 예제 클래스 정의
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]["input_ids"], "attention_mask": self.examples[idx]["attention_mask"]}

# Fine-tuning 데이터 예제 생성
train_examples = [
    InputExample(texts=["question 1", "context 1 for question 1"]),
    InputExample(texts=["question 2", "context 2 for question 2"]),
    # 추가적인 fine-tuning 데이터 예제를 여기에 추가
]

# SentenceTransformer 모델 로드
model_name = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Fine-tuning을 위한 데이터로더 생성
train_dataset = CustomDataset(model.create_batch_hard_triplet_sampler(train_examples))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fine-tuning을 위한 Trainer 설정
num_epochs = 3
warmup_steps = 100
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * num_epochs)
loss_function = losses.TripletMarginLoss(margin=0.5)
trainer = SentenceTransformer.Trainer(model, train_dataloader, optimizer, scheduler=scheduler, loss_function=loss_function, show_progress_bar=True, num_epochs=num_epochs)

# Fine-tuning 진행
trainer.train()

# Fine-tuned 모델 저장
model.save("fine_tuned_model")
