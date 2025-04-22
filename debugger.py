import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import deepspeed


#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", use_fast=False,device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
texts = ["Hello, DeepSpeed!", "DeepSpeed makes large model training efficient."]

# 4ビット量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",quantization_config=bnb_config,device_map="auto")
print('ok!')

