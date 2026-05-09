import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

text = "123\n2 2\n\n2　2,2.2，2。2你好"
tokens = tokenizer(text)
print(tokens)
print(tokenizer.decode(tokens["input_ids"]))