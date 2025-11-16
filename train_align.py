import os
import argparse
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,DynamicCache,AutoConfig
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
from transformers import get_cosine_schedule_with_warmup,get_scheduler
import json
import pandas as pd
import re
import signal
import sys
from sys import exit

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

print(f"----------PID: {os.getpid()}-*---------")


parser=argparse.ArgumentParser()
parser.add_argument("--model_dir",type=str)
parser.add_argument("--adapter_path",type=str)
parser.add_argument("--version_name",type=str,default="qwen2_5vl_reproduce_align")
parser.add_argument("--log_dir",type=str,)
parser.add_argument("--saved_model_dir",type=str)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
parser.add_argument("--sample_num", type=int, default=100, help="Sampling number")
parser.add_argument("--forward_nums", type=int, default=4, help="Forward numbers")
parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
parser.add_argument("--top_k", type=int, default=1, help="Top-k value")
parser.add_argument("--coco_path",type=str)
parser.add_argument("--sharegpt4v_path",type=str)
parser.add_argument("--sharegpt4v_caption_path",type=str)
parser.add_argument("--videor1_cot_dataset",type=str)
parser.add_argument("--videor1_cot_165k",type=str)
args=parser.parse_args()

batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
accumulation_steps = args.accumulation_steps
warmup_ratio = args.warmup_ratio
sample_num = args.sample_num
forward_nums = args.forward_nums
weight_decay = args.weight_decay
top_k = args.top_k

model_dir=args.model_dir
version_name=args.version_name

coco_path=args.coco_path

sharegpt4v_caption_path=args.sharegpt4v_caption_path
videor1_cot_dataset=args.videor1_cot_dataset
videor1_cot_165k=args.videor1_cot_165k

if args.log_dir is None:
    log_dir=str(Path(__file__).parent / version_name)
if args.saved_model_dir is None:
    saved_model_dir=str(Path(__file__).parent / version_name)

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print(version_name,os.getenv('CUDA_VISIBLE_DEVICES'))

config=AutoConfig.from_pretrained(model_dir)
target_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype='auto',config=config).eval()

config.rope_scaling=None
model=Model(config, target_model=target_model)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)


model.load_model(args.adapter_path)
model.draft_model.init_draft_decode_router()
model=model.cuda()
model.device='cuda'

count=0
for param in model.parameters():
    if param.requires_grad==True:

        count+=param.numel()
        


new_dataset=[]

with open(videor1_cot_165k, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

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
        


with open(sharegpt4v_caption_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
        
for example in dataset:
    example={"image":f'{example['id']}.jpg',
                            "prompt": example['conversations'][0]['value'],
                            "caption": example['conversations'][1]['value']}
    
    image_path=os.path.join(coco_path,example['image'])
    
    prompt=[
        {
            "role":"system",
            "content":"You are a helpful assistant."
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
                    "text": re.sub('<image>','',example['prompt'])
                }
            ],
        }
    ]
    
    new_dataset.append({
        'prompt':prompt,
        'image_path':str(image_path),
        'answer':str(example['caption'])
    })
    
dataset=new_dataset


class DataCollator:
    def __init__(self, processor):
        self.processor=processor
        
    def __call__(self, batch):
        
        assert len(batch)==1
        
        prompt_length=[]
        
        example=batch[0]
        
        image_path=example['image_path']
 
        if not os.path.exists(image_path):
            return {
                'inputs':None,
                'prompt_length':None
            }
        
        message=example['prompt']+[
            {
                'role':'assistant',
                'content':example['answer']
            }
        ]
        
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=text, images=image_inputs, videos=video_inputs, max_length=2048, truncation=True,
            padding='longest',return_tensors="pt",
        )

            
        message=example['prompt']
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        cur_inputs = processor(
            text=text, images=image_inputs, videos=video_inputs,
            padding='longest',return_tensors="pt",
        )
        
        prompt_length.append(cur_inputs.input_ids.shape[-1])
            
        return {
            'inputs':inputs,
            'prompt_length':prompt_length
        }



def compute_acc(target_logits,draft_logits,valid_positions,k=2):

    target_indices = torch.argmax(target_logits, dim=-1)
    draft_topk_values, draft_topk_indices = torch.topk(draft_logits, k=k, dim=-1)

    top1_hit = draft_topk_indices[..., 0] == target_indices               
    topk_hit = (draft_topk_indices == target_indices.unsqueeze(-1)).any(dim=-1)  

    correct_top1 = (top1_hit & valid_positions).sum().item()
    correct_topk = (topk_hit & valid_positions).sum().item()
    total_valid_tokens = valid_positions.sum().item()
    
    return correct_top1,correct_topk,total_valid_tokens

def compute_normalized_gradient_l2_norm(model):
    gradient_l2_norm = torch.norm(
        torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    )
    num_grad_params = sum(
        param.grad.numel() for param in model.parameters() if param.grad is not None
    )
    normalized_gradient_l2_norm = gradient_l2_norm / num_grad_params
    
    return normalized_gradient_l2_norm

datacollator=DataCollator(processor)
dataloader=DataLoader(dataset,collate_fn=datacollator,num_workers=8,persistent_workers=True,batch_size=batch_size,shuffle=True,drop_last=False)


optimizer = torch.optim.AdamW(model.draft_model.parameters(), lr=lr)
l1_loss=nn.SmoothL1Loss(reduction='none')

num_training_steps = num_epochs * ((len(dataloader)+accumulation_steps-1)//accumulation_steps)
num_warmup_steps = int(warmup_ratio * num_training_steps)
print(num_training_steps)
lr_scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    scheduler_specific_kwargs={'min_lr_rate':0.2}, 
)

total_correct_top1=[]
total_correct_topk=[]
total_token_nums=[]

step=0
accumulated_step=0
batch_logs=[]
start_time=time.time()

for epoch in tqdm(range(num_epochs)):

    log_file = log_dir + f"/epoch_{epoch}.log"
    with open(log_file,'w',encoding='utf-8') as f:
        pass

    for i,batch in enumerate(dataloader):
        
        inputs=batch['inputs'].to('cuda')
        prompt_length=batch['prompt_length'][0]
        
        if inputs is None:
            continue
        
        if prompt_length>=inputs.input_ids.shape[-1]:
            continue
        
        with torch.no_grad():
            target_outputs=model.target_model(**inputs,output_hidden_states=True)

        last_hidden_states=target_outputs.hidden_states[-1]
        feature_states=last_hidden_states
        target_logits=target_outputs.logits
        
        prefill_feature_states=feature_states[:,:prompt_length-1,:]
        prefill_attention_mask=inputs.attention_mask[:,:prompt_length-1]
        
        decode_feature_states=feature_states[:,prompt_length-1:-1,:] 
        decode_attention_mask=inputs.attention_mask[:,:-1] 
        decode_input_ids=inputs.input_ids[:, prompt_length:] 

        draft_outputs=model(hidden_states=prefill_feature_states, input_ids=None,
                            router=1, attention_mask=prefill_attention_mask)
        past_key_values=draft_outputs['past_key_values']
        
        with torch.no_grad():
            target_logits=target_logits[:,prompt_length:,:].float().softmax(dim=-1).detach()
        
        past_hidden_states = None
        cur_loss=None
        loss_mask=torch.ones((decode_feature_states.shape[0], decode_feature_states.shape[1]), device=decode_feature_states.device)
        
        for forward_idx in range(forward_nums):
            router=2 if forward_idx==0 else 3
            
            draft_outputs = model(hidden_states=decode_feature_states, input_ids=decode_input_ids,
                            router=router, attention_mask=decode_attention_mask, past_key_values=past_key_values,
                            use_align=True, forward_num=forward_idx+1, past_hidden_states=past_hidden_states)
            
            next_feature_states=draft_outputs['next_feature_states']
            draft_hidden_states=draft_outputs['hidden_states']

            if past_hidden_states is None:
                past_hidden_states = torch.cat([decode_feature_states[:, :1, :], next_feature_states[:, :-1, :]], dim=1)[None, :, :, :]
            else:
                new_past_hidden_states = torch.cat([decode_feature_states[:, :1, :], next_feature_states[:, :-1, :]], dim=1)[None, :, :, :]
                past_hidden_states = torch.cat([past_hidden_states, new_past_hidden_states], dim=0)
            past_hidden_states = past_hidden_states.detach()
        
            draft_logits=model.lm_head(draft_hidden_states)
            
            loss1=l1_loss(next_feature_states,feature_states[:,prompt_length:,:])

            loss1=torch.mean(loss1,dim=-1)*loss_mask
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask, dim=-1)
            loss1=loss1.mean()
            loss1=loss1*2
            
            draft_logits=draft_logits.float().softmax(dim=-1)
            
            plogp=target_logits*torch.log(draft_logits)
            loss2=torch.sum(plogp,dim=-1)*loss_mask
            loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask, dim=-1)
            loss2= - loss2.mean()

            loss2=loss2*0.1
        
            loss=loss1+loss2
            loss/=accumulation_steps
            
            topk_values, topk_tokens = torch.topk(draft_logits, k=top_k, dim=-1)  

            target_tokens = decode_input_ids[:, 1:].unsqueeze(-1)  
            pred_topk = topk_tokens[:, :-1, :]  

            loss_mask = (pred_topk == target_tokens).any(dim=-1).float() 

            loss_mask = torch.concat([
                torch.ones((loss_mask.shape[0], 1), device=loss_mask.device, dtype=loss_mask.dtype),
                loss_mask], dim=-1)

            if cur_loss is None:
                cur_loss=loss
            else:
                cur_loss=cur_loss+loss*(weight_decay**forward_idx)

            valid_positions=torch.ones(target_logits.shape[:-1],dtype=torch.bool,device=model.target_model.device)
            with torch.no_grad():
                correct_top1,correct_topk,total_valid_tokens=compute_acc(target_logits,draft_logits,valid_positions,k=4)
            
            total_correct_top1.append(correct_top1)
            total_correct_topk.append(correct_topk)
            total_token_nums.append(total_valid_tokens)

            batch_logs.append({
                'loss': loss.item(),
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'correct_top1': correct_top1,
                'correct_topk': correct_topk,
                'total_valid_tokens': total_valid_tokens,
                'forward_idx': forward_idx 
            })
               
        cur_loss.backward()
        accumulated_step+=1

        
        if accumulated_step%accumulation_steps==0:
            
            step+=1
            real_sample_num=sample_num*accumulation_steps

            per_forward_logs = []
            for idx in range(forward_nums):
                idx_mask = [log for log in batch_logs if 'forward_idx' in log and log['forward_idx'] == idx]
                if not idx_mask:
                    continue
                loss = sum(log["loss"] for log in idx_mask)
                loss1 = sum(log["loss1"] for log in idx_mask)
                loss2 = sum(log["loss2"] for log in idx_mask)
                correct_top1 = sum(log["correct_top1"] for log in idx_mask)
                correct_topk = sum(log["correct_topk"] for log in idx_mask)
                total_valid_tokens = sum(log["total_valid_tokens"] for log in idx_mask)

                top1_acc = round(correct_top1 / total_valid_tokens, 4) if total_valid_tokens else 0
                topk_acc = round(correct_topk / total_valid_tokens, 4) if total_valid_tokens else 0

                per_forward_logs.append({
                    f"loss_forward_{idx}": round(loss / len(idx_mask), 4),
                    f"loss1_forward_{idx}": round(loss1 / len(idx_mask), 4),
                    f"loss2_forward_{idx}": round(loss2 / len(idx_mask), 4),
                    f"top1_acc_forward_{idx}": top1_acc,
                    f"topk_acc_forward_{idx}": topk_acc
                })

            global_loss = round(sum(log["loss"] for log in batch_logs) / len(batch_logs), 4)
            global_loss1 = round(sum(log["loss1"] for log in batch_logs) / len(batch_logs), 4)
            global_loss2 = round(sum(log["loss2"] for log in batch_logs) / len(batch_logs), 4)

            global_total_tokens = sum(log["total_valid_tokens"] for log in batch_logs)
            global_top1_acc = round(sum(log["correct_top1"] for log in batch_logs) / global_total_tokens, 4) if global_total_tokens else 0
            global_topk_acc = round((sum(log["correct_topk"] for log in batch_logs)) / global_total_tokens, 4) if global_total_tokens else 0

            last_sample_total_tokens = sum(total_token_nums[-real_sample_num:])
            last_sample_top1 = round(sum(total_correct_top1[-real_sample_num:]) / last_sample_total_tokens, 4) if last_sample_total_tokens else 0
            last_sample_topk = round(sum(total_correct_topk[-real_sample_num:]) / last_sample_total_tokens, 4) if last_sample_total_tokens else 0

            avg_logs = {
                "step": step,
                "used_time": round((time.time() - start_time) / 60, 4),
                "global_loss": global_loss,
                "global_loss1": global_loss1,
                "global_loss2": global_loss2,
                "global_top1_acc": global_top1_acc,
                "global_topk_acc": global_topk_acc,
                f"last{sample_num}_top1_acc": last_sample_top1,
                f"last{sample_num}_topk_acc": last_sample_topk,
            }

            for log_dict in per_forward_logs:
                avg_logs.update(log_dict)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_logs) + '\n')
                
            batch_logs.clear()

            total_correct_top1=total_correct_top1[-real_sample_num:]
            total_correct_topk=total_correct_topk[-real_sample_num:]
            total_token_nums=total_token_nums[-real_sample_num:]
                
            batch_logs.clear()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step%8000==0 and step!=0:
                model.save_model(f'{saved_model_dir}/step{step}.pth')
            
            if (step*accumulation_steps)%16==0:
                torch.cuda.empty_cache()
    

model.save_model(f'{saved_model_dir}/step{step}.pth')  
