import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import deepspeed

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.input_ids = tokenizer(texts, return_tensors='pt', padding=True)['input_ids']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        return {'input_ids': input_id, 'labels': input_id}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
texts = ["Hello, DeepSpeed!", "DeepSpeed makes large model training efficient."]
dataset = SimpleDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size= 2,shuffle=True)

# 4ビット量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = GPT2LMHeadModel.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",quantization_config=bnb_config)
model = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16)


# Initialize DeepSpeed
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters = model.parameters(),
    config = 'deepspeed_config.json'
)

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        model.backward(loss)
        model.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')



# トレーニング処理（DeepSpeedなど）が終了した後に追加
if dist.is_initialized():  # プロセスグループが初期化されている場合のみ実行
    dist.destroy_process_group()
