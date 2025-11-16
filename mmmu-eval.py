import ast
import argparse
import sys
from sys import exit
import shutil
from pathlib import Path
import io
import json
import base64
import re
import os
from typing import Optional, Union
import  time
import pandas as pd
from PIL import Image


from tqdm import tqdm
import torch
from datasets import Dataset,concatenate_datasets,disable_progress_bar,enable_progress_bar
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
)
from qwen_vl_utils import process_vision_info

from helper.modeling_draft import Model
from helper.speculative_generate_vl import speculative_generate


print(f"************PID: {os.getpid()}************")

parser=argparse.ArgumentParser()

parser.add_argument("--model_path", type=str,)
parser.add_argument("--dataset_path", type=str,)
parser.add_argument("--adapter_path", type=str,)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--draft_length", type=int, default=5)
parser.add_argument("--draft_k", type=int, default=8)
parser.add_argument("--draft_total_token", type=int, default=63)
parser.add_argument("--do_sample", action="store_true",)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--sample_num", type=int, default=10)
parser.add_argument("--min_pixels", type=int, default=256*28*28)
parser.add_argument("--max_pixels", type=int, default=1280*28*28)

args=parser.parse_args()



model_path=args.model_path
dataset_path=Path(args.dataset_path)
adapter_path=args.adapter_path
batch_size=args.batch_size
draft_length=args.draft_length
draft_k=args.draft_k
draft_total_token=args.draft_total_token
do_sample=args.do_sample
num_return_sequences=args.num_return_sequences
sample_num=args.sample_num

config = AutoConfig.from_pretrained(model_path)

min_pixels = 256*28*28
max_pixels = 1280*28*28


target_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype='auto',device_map='cuda').eval()

config.rope_scaling=None
model=Model(config,target_model=target_model)

model.draft_model.init_draft_decode_router()      
model.load_model(adapter_path)
model=model.cuda()
model.device='cuda'

processor=AutoProcessor.from_pretrained(model_path,min_pixels=min_pixels,max_pixels=max_pixels,padding_side='left')

dataset_files = list(dataset_path.glob("**/dev*.parquet")) + list(dataset_path.glob("**/validation*.parquet"))


all_datasets = []
disable_progress_bar()
for file in tqdm(dataset_files):
    if file.exists():
        dataset=Dataset.from_parquet(str(file))
        all_datasets.append(dataset)
enable_progress_bar()
dataset = concatenate_datasets(dsets=all_datasets)
dataset = dataset.filter(lambda x: x["image_2"] is None)

system_prompt = (
    "You are a helpful assistant. You will be given a question later."
    "The user asks a question, and you solves it. "
    "You first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

template_data = []
print("\nPreprocessing......")
for data in tqdm(dataset):
    options = ast.literal_eval(data["options"])
    options = [f"{chr(index + ord('A'))}. {item}" for index, item in enumerate(options)]
    formatted_option_text = "\n".join(options)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data["image_1"]},
                {
                    "type": "text",
                    "text": f"{data['question']} Please choose a best option to this question:\n"
                    + formatted_option_text
                    + '\nPlease present your thinking process (enclosed in the format "<think> ... </think>")'
                    'and a final answer (a letter enclosed in the formmat "<answer> ... </answer>").',
                },
            ],
        },
    ]
    template_data.append(messages)

dataset_size=len(dataset)          


total_length=0
total_acc_length=0
total_token_num=0
total_decoded_token_num=0
total_target_time=0
total_draft_time=0
total_check_time=0
total_prefill_time=0
total_post_time=0
total_speculative_time_cost=0

plain_inference_time=0
plain_inference_generated_max_token_num=0
plain_inference_total_generated_token_num=0

for i in tqdm(range(0,dataset_size,batch_size)):
    batch_template=template_data[i:i+batch_size]
    batch_labels=dataset[i:i+batch_size]["answer"] 

    batch_text = []
    batch_images = []
    for template in batch_template:
        text = processor.apply_chat_template(template,tokenize=False,add_generation_prompt=True)
        batch_text.append(text)

        image_input, _= process_vision_info(template)
        batch_images.extend(image_input)
    
    inputs: dict[str,torch.Tensor] = processor(text=batch_text,
                                              images=batch_images,
                                              padding='longest',
                                              return_tensors='pt')
    inputs = inputs.to(model.device)

    inference_start_time =  time.perf_counter()
    with  torch.inference_mode():       
        output_ids:torch.Tensor = model.target_model.generate(**inputs,
                                       max_new_tokens=4096,
                                       do_sample=do_sample)
    inference_end_time = time.perf_counter()
    plain_inference_time += (inference_end_time-inference_start_time)   

    generated_ids:list[torch.Tensor] = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids,output_ids)]
    
    plain_inference_generated_max_token_num += max([seq.numel() for seq in generated_ids])
    plain_inference_total_generated_token_num+=sum([seq.numel() for seq in generated_ids])

    with torch.inference_mode():
        outputs=speculative_generate(model,inputs,draft_length,do_sample,1024,num_return_sequences,1.0,0.95,draft_k=draft_k,
                                return_all_draft_input=True,vocab_map=None,draft_total_token=draft_total_token)
    decoded_sequences=[processor.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]

    total_length+=outputs['max_sequence_length']
    total_acc_length+=outputs['total_acc_length']
    total_decoded_token_num+=outputs['total_decoded_token_num']
    total_target_time+=outputs['target_time_cost']
    total_draft_time+=outputs['draft_time_cost']
    total_check_time+=outputs['check_time_cost']
    total_prefill_time+=outputs['prefill_time_cost']
    total_post_time+=outputs['post_time_cost']
    total_speculative_time_cost+=outputs["total_time_cost"]

print("------MMMU------")
print(f"{adapter_path=}")
print(f"{batch_size=}")
print(f"{draft_length=}")
print(f"{draft_k=}")
print(f"{draft_total_token=}")
print(f"{do_sample=}")
print(f"{num_return_sequences=}")
print(f"{sample_num=}")
print(f"{dataset_size=}")

print("\n----------\n")

print(f"tps (plain inference): {plain_inference_generated_max_token_num/plain_inference_time}")
print(f"{plain_inference_time=}")
print(f"tps (speculative): {total_length/total_speculative_time_cost}")
print(f"{total_acc_length/total_decoded_token_num=}")
print(f"{total_target_time=}   {total_draft_time=}   {total_check_time=}")
print(f"{total_prefill_time=}   {total_post_time=}")
print(f"{total_speculative_time_cost=}")
print(f"Average generated token account: {plain_inference_total_generated_token_num/dataset_size}")