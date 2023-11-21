from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("/prj/src/out/berttoken/", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
encoded = tokenizer.encode("왜이렇게 잘 안되는거야")

print(encoded)