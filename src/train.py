from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from modules import Custom_Tokenizer



use_flash_attention = False
#databricks/databricks-dolly-15k
dataset = load_dataset("royboy0416/ko-alpaca", split="train")

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])

#train tokenizer
#transformers: "../out/bertvocab/vocab.json"
#bert: '../out/bertvocab/vocab.txt'
custom_token = Custom_Tokenizer(vocab_dir="sentencebpevocab/test.json", token_dir="sentencebpetoken/",
                                vocab_file_name="prepare.mecab.txt", raw_data_url="https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",
                                save_raw_name="ratings.txt", model_name="sentencebpe")

custom_token.run()

def format_instruction(sample):
    return f"""### Instruction:
    아래 input으로 LLM을 사용하여 입력을 생성하는 데 사용될 수 있는 명령을 생성하십시오.

    ### input:
    {sample['output']}

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
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# print(tokenizer)

#tokenizer를 가져올때 상대경로가 아닌 절대경로로 해야 잘된다
tokenizer = AutoTokenizer.from_pretrained("/prj/src/out/sentencebpetoken/")
tokenizer.pad_token_id = 0
tokenizer.eos_token_id = 1
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

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args
)

trainer.train()

trainer.save_model()