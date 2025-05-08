import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.input_ids = tokenizer(texts, return_tensors='pt', padding=True)['input_ids']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        return {'input_ids': input_id, 'labels': input_id}

def init_distributed():
    """分散処理の初期化"""
    deepspeed.init_distributed()

    # デバイスの設定
    torch.cuda.set_device(args.local_rank)

    # 環境変数の設定
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )


# メインコードの修正
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSpeed training script")
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--mp_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()

    # 分散処理の初期化
    init_distributed()

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザーの設定
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    try:
    # モデルの初期化
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None  # 'auto'から変更
        )
        print('Pretrained model is loaded successfully')
        # DeepSpeedの設定
        ds_config = {
            "train_batch_size": 2,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
            "distributed_backend": "nccl"
        }

        # DeepSpeedの初期化
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config='deepspeed_config.json'
        )

        print('DeepSpeed model is initialized successfully')
        # データセットとデータローダーの設定
        texts = ["Hello, DeepSpeed!", "DeepSpeed makes large model training efficient."]
        dataset = SimpleDataset(tokenizer, texts)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 学習ループ
        for epoch in range(3):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(model.device)
                labels = batch['labels'].to(model.device)
                outputs = model(input_ids, labels=labels)
                loss = outputs[0]
                model.backward(loss)
                model.step()

                if args.local_rank == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    finally:
        # クリーンアップ
        if dist.is_initialized():
            dist.destroy_process_group()