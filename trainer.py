import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import deepspeed
import argparse

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.input_ids = tokenizer(texts, return_tensors='pt', padding=True)['input_ids']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        return {'input_ids': input_id, 'labels': input_id}

# コマンドライン引数を定義
parser = argparse.ArgumentParser(description="DeepSpeed training script")
# DeepSpeed用に引数を解析
parser = deepspeed.add_config_arguments(parser)  # DeepSpeed関連の引数を追加
parser.add_argument("--mp_size", type=int, default=1, help="Model parallelism size")
parser.add_argument("--local_rank", type=int, default=0, help="Model parallelism size")
parser.add_argument("--model_name", type=str, default="gpt2", help="DeepSpeed config file path")
# 引数を解析
args = parser.parse_args()

print(f"Local rank: {args.local_rank}")
print(f"Model parallelism size: {args.mp_size}")

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,device_map="auto")
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

model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
                device_map="auto",  # 여러 GPU에 자동 분산
                    torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                        )
print('pretrained model load ok!')
model = deepspeed.init_inference(
                model,
                mp_size=args.mp_size,  # 사용 가능한 GPU 수에 맞게 조정
                dtype=torch.float16,
                replace_method="auto",
                replace_with_kernel_inject=False,
            )

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
