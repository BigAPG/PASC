import os
from sys import exit
import argparse
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, DynamicCache, AutoConfig
from qwen_vl_utils import process_vision_info

from helper.modeling_draft import Model
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
import datasets
from transformers import get_cosine_schedule_with_warmup, get_scheduler
import json
import pandas as pd
import re
import signal
import sys

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, )
parser.add_argument("--adapter_path", type=str, default=None)
parser.add_argument("--version_name", type=str, default='qwen2_5vl_reproduce')
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--saved_model_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
parser.add_argument("--sample_num", type=int, default=100, help="Sampling number")
parser.add_argument("--num_workers", type=int, default=1, help="DataLoader num_workers")
parser.add_argument("--sharegpt4v_path", type=str, )
parser.add_argument("--coco_path", type=str, )
parser.add_argument("--videor1_cot_165k",type=str)
parser.add_argument("--videor1_cot_dataset",type=str)

args = parser.parse_args()

batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
accumulation_steps = args.accumulation_steps
warmup_ratio = args.warmup_ratio
sample_num = args.sample_num
num_workers = args.num_workers

model_dir = args.model_dir
version_name = args.version_name
adapter_path = args.adapter_path

sharegpt4v_path = args.sharegpt4v_path
coco_path = args.coco_path
videor1_cot_165k=args.videor1_cot_165k
videor1_cot_dataset=args.videor1_cot_dataset

if args.log_dir is None:
    log_dir = str(Path(__file__).parent / version_name)
else:
    log_dir = args.log_dir

if args.saved_model_dir is None:
    saved_model_dir = str(Path(__file__).parent / version_name)
else:
    saved_model_dir = args.saved_model_dir

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir, exist_ok=True)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

print(version_name, os.getenv('CUDA_VISIBLE_DEVICES'))

config = AutoConfig.from_pretrained(model_dir)
target_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype='auto', config=config).eval()

config.rope_scaling = None
model = Model(config, target_model=target_model)


if adapter_path is not None:
    try:
        model.load_model(adapter_path)

        if hasattr(model.draft_model, "init_draft_decode_router"):
            try:
                model.draft_model.init_draft_decode_router()
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: load_model failed for {adapter_path}: {e}")

model = model.cuda()

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)


count = 0
for param in model.parameters():
    if param.requires_grad == True:
        print(param.shape)
        count += param.numel()

print(count/1000/1000, 'M')

with open(videor1_cot_165k, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
    

new_dataset = []
for example in dataset:
    if example['data_type']=='image':
        
        image_path=os.path.join(videor1_cot_dataset, example['path'])

        instruction=example['problem']+'\n'+'Options:'+'\n'
        for option in example['options']:
            instruction=instruction+option+'\n'
        answer=example['process']+example['solution']
        
        prompt=[
            {
                "role" : "system",
                "content" : "You are a helpful assistant."
            } , 
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request."
                        "Your response should include your thought process enclosed within <think></think> tags" 
                        "and the final answer enclosed within <answer></answer> tags.\n"
                        f"### Instruction:\n{instruction}"
                    }
                ]
            }
        ] 
        
        new_dataset.append({
            'prompt': prompt,
            'image_path':str(image_path),
            'answer':str(answer)
        })
        
with open(sharegpt4v_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

for example in dataset:
    example = {"image": f'{example["id"]}.jpg',
               "prompt": example['conversations'][0]['value'],
               "caption": example['conversations'][1]['value']}
    image_path = str(Path(coco_path) / example["image"])
    prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": re.sub('<image>', '', example['prompt'])
                }
            ],
        }
    ]

    new_dataset.append({
        'prompt': prompt,
        'image_path': str(image_path),
        'answer': str(example['caption'])
    })

dataset = new_dataset


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        assert len(batch) == 1

        prompt_length = []

        example = batch[0]

        image_path = example['image_path']

        if not os.path.exists(image_path):
            return {
                'inputs': None,
                'prompt_length': None
            }

        message = example['prompt'] + [
            {
                'role': 'assistant',
                'content': example['answer']
            }
        ]

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=text, images=image_inputs, videos=video_inputs, max_length=2048, truncation=True,
            padding='longest', return_tensors="pt",
        )

        message = example['prompt']
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        cur_inputs = self.processor(
            text=text, images=image_inputs, videos=video_inputs,
            padding='longest', return_tensors="pt",
        )

        prompt_length.append(cur_inputs.input_ids.shape[-1])

        return {
            'inputs': inputs,
            'prompt_length': prompt_length
        }


def add_gaussian_noise(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.01
) -> torch.Tensor:

    noise = torch.randn_like(tensor) * std + mean

    tensor = tensor + noise

    return tensor


def add_uniform_noise(tensor, std=0.05):

    noise = (torch.rand_like(tensor) - 0.5) * std
    tensor = tensor + noise

    return tensor


def compute_acc(target_logits, draft_logits, valid_positions, k=2):

    target_indices = torch.argmax(target_logits, dim=-1)
    draft_topk_values, draft_topk_indices = torch.topk(draft_logits, k=k, dim=-1)

    top1_hit = draft_topk_indices[..., 0] == target_indices
    topk_hit = (draft_topk_indices == target_indices.unsqueeze(-1)).any(dim=-1)

    correct_top1 = (top1_hit & valid_positions).sum().item()
    correct_topk = (topk_hit & valid_positions).sum().item()
    total_valid_tokens = valid_positions.sum().item()

    return correct_top1, correct_topk, total_valid_tokens


def compute_normalized_gradient_l2_norm(model):
    gradient_l2_norm = torch.norm(
        torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    )
    num_grad_params = sum(
        param.grad.numel() for param in model.parameters() if param.grad is not None
    )
    normalized_gradient_l2_norm = gradient_l2_norm / num_grad_params

    return normalized_gradient_l2_norm


datacollator = DataCollator(processor)
dataloader = DataLoader(dataset, collate_fn=datacollator, num_workers=num_workers, persistent_workers=True,
                        batch_size=batch_size, shuffle=True, drop_last=False)


optimizer = torch.optim.AdamW(model.draft_model.parameters(), lr=lr)
l1_loss = nn.SmoothL1Loss(reduction='none')

num_training_steps = num_epochs * ((len(dataloader) + accumulation_steps - 1) // accumulation_steps)
num_warmup_steps = int(warmup_ratio * num_training_steps)
print(num_training_steps)
lr_scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    scheduler_specific_kwargs={'min_lr_rate': 0.2},
)

total_correct_top1 = []
total_correct_topk = []
total_token_nums = []

step = 0
accumulated_step = 0
batch_logs = []
start_time = time.time()

for epoch in range(num_epochs):

    log_file = log_dir + f"/epoch_{epoch}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        pass

    for i, batch in enumerate(dataloader):
        inputs = batch['inputs'].to('cuda')
        prompt_length = batch['prompt_length'][0]

        if inputs is None:
            continue

        if prompt_length >= inputs.input_ids.shape[-1]:
            continue

        with torch.no_grad():
            target_outputs = model.target_model(**inputs, output_hidden_states=True)

        last_hidden_states = target_outputs.hidden_states[-1]
        feature_states = last_hidden_states
        target_logits = target_outputs.logits

        prefill_feature_states = feature_states[:, :prompt_length-1, :]  
        prefill_attention_mask = inputs.attention_mask[:, :prompt_length-1]

        decode_feature_states = feature_states[:, prompt_length-1:-1, :]  
        decode_attention_mask = inputs.attention_mask[:, :-1]  
        decode_input_ids = inputs.input_ids[:, prompt_length:]
        decode_feature_states_with_noise = decode_feature_states

        draft_outputs = model(hidden_states=prefill_feature_states, input_ids=None,
                              router=1, attention_mask=prefill_attention_mask)
        past_key_values = draft_outputs['past_key_values']

        draft_outputs = model(hidden_states=decode_feature_states_with_noise, input_ids=decode_input_ids,
                              router=2, attention_mask=decode_attention_mask, past_key_values=past_key_values)

        next_feature_states = draft_outputs['next_feature_states']
        draft_hidden_states = draft_outputs['hidden_states']
        draft_logits = model.lm_head(draft_hidden_states)


        loss1 = l1_loss(next_feature_states, feature_states[:, prompt_length:, :])

        loss1 = torch.mean(loss1, dim=-1)
        loss1 = loss1.mean()
        loss1 = loss1 * 2


        with torch.no_grad():
            target_logits = target_logits[:, prompt_length:, :].float().softmax(dim=-1).detach()
        draft_logits = draft_logits.float().softmax(dim=-1)

        plogp = target_logits * torch.log(draft_logits)
        loss2 = torch.sum(plogp, dim=-1)
        loss2 = - loss2.mean()

        loss2 = loss2 * 0.1

        loss = loss1 + loss2

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            loss = loss.detach()
            del loss
            del feature_states, next_feature_states, target_logits, draft_logits
            torch.cuda.empty_cache()
        else:
            accumulated_step += 1


            if accumulated_step % accumulation_steps == 1: 
                optimizer.zero_grad(set_to_none=True)
                loss2.backward(retain_graph=True)
                loss2_norm = compute_normalized_gradient_l2_norm(model.draft_model.arlayer)
                optimizer.zero_grad(set_to_none=True)

                loss1.backward(retain_graph=True)
                loss1_norm = compute_normalized_gradient_l2_norm(model.draft_model.arlayer)
                optimizer.zero_grad(set_to_none=True)

            loss /= accumulation_steps
            loss.backward()

            valid_positions = torch.ones(target_logits.shape[:-1], dtype=torch.bool, device=model.target_model.device)
            with torch.no_grad():
                correct_top1, correct_topk, total_valid_tokens = compute_acc(target_logits, draft_logits, valid_positions, k=4)

            total_correct_top1.append(correct_top1)
            total_correct_topk.append(correct_topk)
            total_token_nums.append(total_valid_tokens)

            batch_logs.append({
                'loss': loss.item() * accumulation_steps,
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'loss1_norm': loss1_norm.item(),
                'loss2_norm': loss2_norm.item(),
                'correct_top1': correct_top1,
                'correct_topk': correct_topk,
                'total_valid_tokens': total_valid_tokens
            })

            if accumulated_step % accumulation_steps == 0:
                step += 1
                real_sample_num = sample_num * accumulation_steps

                avg_logs = {
                    "step": step,
                    "loss": round(sum(log["loss"] for log in batch_logs) / len(batch_logs), 4),
                    "used_time": round((time.time() - start_time) / 60, 3),
                    "loss1": round(sum(log["loss1"] for log in batch_logs) / len(batch_logs), 4),
                    "loss2": round(sum(log["loss2"] for log in batch_logs) / len(batch_logs), 4),
                    "loss1_norm": sum(log["loss1_norm"] for log in batch_logs),
                    "loss2_norm": sum(log["loss2_norm"] for log in batch_logs),
                    "top1_acc": round(sum(log['correct_top1'] for log in batch_logs) / sum(log['total_valid_tokens'] for log in batch_logs), 4),
                    "topk_acc": round(sum(log['correct_topk'] for log in batch_logs) / sum(log['total_valid_tokens'] for log in batch_logs), 4),
                    f"last{sample_num}_top1_acc": round(sum(total_correct_top1[-real_sample_num:]) / sum(total_token_nums[-real_sample_num:]), 4),
                    f"last{sample_num}_topk_acc": round(sum(total_correct_topk[-real_sample_num:]) / sum(total_token_nums[-real_sample_num:]), 4),
                }

                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(avg_logs) + '\n')

                total_correct_top1 = total_correct_top1[-real_sample_num:]
                total_correct_topk = total_correct_topk[-real_sample_num:]
                total_token_nums = total_token_nums[-real_sample_num:]

                batch_logs.clear()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if step % 8000 == 0 and step != 0:
                    model.save_model(f'{saved_model_dir}/step{step}.pth')

                if (step * accumulation_steps) % 16 == 0:
                    torch.cuda.empty_cache()

model.save_model(f'{saved_model_dir}/step{step}.pth')
