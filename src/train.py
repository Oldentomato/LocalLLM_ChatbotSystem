from darasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTrainer

use_flash_attention = False

dataset = load_dataset("royboy0416/ko-alpaca", split="train")

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])

def format_instruction(sample):
    return f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 

    ### Input:
    {sample['response']}

    ### Response:
    {sample['instruction']}
    """

print(format_instruction(dataset[randrange(len(dataset))]))

model_id = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config = bnb_config,
    use_cache = False,
    use_flash_attention_2 = use_flash_attention,
    device_map="auto"
)

model.config.pretraining_tp = 1

#이 부분을 로컬 토크나이저로 변경할 것
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)

args = TrainingArguments(
    output_dir="./out/model/llama-7-int4-dolly",
    num_train_epochs=3,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,  # disable tqdm since with packing values are in correct
)


if use_flash_attention:
    from utils.llama_patch import upcast_layer_for_flash_attention
    torch_dtype= torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = upcast_layer_for_flash_attention(model, torch_dtype)

model = get_peft_model(model, peft_config)

max_seq_length = 2048

trainer = SFTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_fuc=format_instruction,
    args=args
)

trainer.train()

trainer.save_model()