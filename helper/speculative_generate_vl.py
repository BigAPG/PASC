import pandas as pd
from transformers import AutoTokenizer, AutoProcessor,AutoConfig,AutoModelForCausalLM
import torch
import datasets

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
import datasets
from transformers import get_cosine_schedule_with_warmup
from transformers import DynamicCache
import json
import pandas as pd
import re
import signal
import sys
import torch
from copy import deepcopy,copy
import threading
import math
import warnings
from concurrent.futures import ThreadPoolExecutor


total_target_time=0
total_draft_time=0
total_check_time=0


def sampling(
    logits, 
    top_k=None, 
    top_p=None, 
    temperature=0.6, 
    eos_token_id=2
):

    assert logits.dim() == 3, f"Expected logits to have shape [bsz, seq, vocab], got {logits.shape}"
    bsz, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.view(-1, vocab_size)

    if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
        logits_flat = torch.where(
            torch.isnan(logits_flat) | torch.isinf(logits_flat),
            torch.tensor(float('-inf'), dtype=logits_flat.dtype, device=logits_flat.device),
            logits_flat
        )

    valid_mask = ~torch.isinf(logits_flat).all(dim=-1) 
    if not valid_mask.any():
        warnings.warn("All sequences in the batch are invalid. Returning EOS token IDs as fallback.")
        return torch.full(
            (bsz, seq_len), 
            fill_value=eos_token_id, 
            dtype=torch.long, 
            device=logits.device
        )

    sampled_tokens_flat = torch.full(
        (logits_flat.shape[0], ), 
        fill_value=eos_token_id, 
        dtype=torch.long, 
        device=logits.device
    )

    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() > 0:
        valid_logits = logits_flat[valid_indices] / temperature
        probs = F.softmax(valid_logits, dim=-1)
        
        if top_p:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative_probs > top_p
            mask = torch.roll(mask, shifts=1, dims=-1)
            mask[..., 0] = False 

            sorted_probs.masked_fill_(mask, 0.0)
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        
        if top_k:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)

            probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)

        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled_tokens_flat[valid_indices] = sampled_indices

    sampled_tokens = sampled_tokens_flat.view(bsz, seq_len)
    return sampled_tokens



def speculative_generate(model, inputs, draft_token_length,
                         do_sample=False, max_new_token=1024, repeated_generate_nums=None,
                         temperature=1.0, top_p=0.95, top_k=None, 
                         draft_k=4,draft_total_token=31,
                         statistical_time=True,return_all_draft_input=False,vocab_map=None
                         ):


    class Node:
        
        def __init__(self,depth,input_id):
            
            self.depth=depth
            self.input_id=input_id
            
            

    def draft_generate(model,next_feature_states,draft_hidden_states,draft_past_key_values_tree,
                        draft_token_length,past_position_ids_tensor,padding_positions,
                        draft_k=4,draft_total_token=32,vocab_map=None):
        
        global total_check_time
        
        dtype=model.dtype
        device=model.device
        bsz=draft_hidden_states.shape[0]
        
        node_nums=draft_k+draft_k*draft_k*(draft_token_length-1)
        trees=[[0] * node_nums for _ in range(bsz)] 
        parents_list=[]
        
        total_input_ids=[]
        total_position_ids=[]
        confidences=[]
        
        draft_position_ids=past_position_ids_tensor.unsqueeze(-1).repeat(1, draft_k)
        total_position_ids.append(draft_position_ids)
        

        if vocab_map is not None:
            draft_logits=model.draft_lm_head(draft_hidden_states.to(model.target_model.dtype))
        else:
            draft_logits=model.lm_head(draft_hidden_states.to(model.target_model.dtype))
        draft_logits=draft_logits.softmax(dim=-1)
        
        next_token_values, draft_next_token=torch.topk(draft_logits, k=draft_k, dim=-1)
        draft_confidences=next_token_values.view(bsz, -1)
        
        if vocab_map is not None:
            draft_next_token = vocab_map[draft_next_token]
        
        past_kv_len=draft_past_key_values_tree[0][0].shape[-2]
        init_kv_len=draft_past_key_values_tree[0][0].shape[-2]

        for idx_batch in range(bsz):
            
            for idx_k in range(draft_k):
                node=Node(depth=0, input_id=None)
                
                trees[idx_batch][idx_k]=node
                
        draft_next_token=draft_next_token.view(bsz, -1) 
        bsz, _, hidden_size = next_feature_states.shape
        next_feature_states=next_feature_states.expand(bsz, draft_k, hidden_size)

        total_input_ids.append(draft_next_token) 
        confidences.append(draft_confidences) 
        attention_seen_indices=torch.arange(past_kv_len, past_kv_len+draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1) 
                
        for idx_token in range(1,draft_token_length):
            
            draft_position_ids=draft_position_ids+1
            total_position_ids.append(draft_position_ids.repeat(1, draft_k))
            
            min_dtype=torch.finfo(dtype).min
            q_length=draft_k
            
            draft_attention_mask=torch.zeros((q_length, past_kv_len+q_length),dtype=dtype,device=device)
            draft_attention_mask[..., init_kv_len:]=min_dtype
            
            draft_attention_mask=draft_attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1)

            zeros=torch.zeros((bsz, 1, q_length, idx_token), dtype=dtype, device=device)

            draft_attention_mask.scatter_(dim=-1, index=attention_seen_indices.unsqueeze(1), src=zeros)

            if isinstance(padding_positions, torch.Tensor):
                padding_positions_tensor=padding_positions
            else:
                padding_positions_indices=[]
        
                for batch_id, pad_positions in enumerate(padding_positions):
                    for pos in pad_positions:
                        padding_positions_indices.append([batch_id, pos])

                if padding_positions_indices:  

                    padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
                    
                padding_positions_tensor=padding_positions_indices
                
            if isinstance(padding_positions_tensor, torch.Tensor):
                draft_attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype
            torch.cuda.synchronize()
            check_time_start=time.time()
            
            
            if hasattr(model.draft_model, 'draft_decode_mlp'):
                router=3
            else:
                router=2

            draft_outputs=model(hidden_states=next_feature_states, router=router, input_ids=draft_next_token,
                                attention_mask=draft_attention_mask,use_cache=True,
                                past_key_values=draft_past_key_values_tree,position_ids=draft_position_ids)
            
            torch.cuda.synchronize()
            total_check_time+=time.time()-check_time_start 
            
            draft_past_key_values_tree=draft_outputs['past_key_values']
            draft_hidden_states=draft_outputs['hidden_states']
            next_feature_states=draft_outputs['next_feature_states'] 
            

            if vocab_map is not None:
                draft_logits=model.draft_lm_head(draft_hidden_states.to(model.target_model.dtype))
            else:
                draft_logits=model.lm_head(draft_hidden_states.to(model.target_model.dtype))
            draft_logits=draft_logits.softmax(dim=-1)
            
            
            next_token_values,draft_next_token=torch.topk(draft_logits, k=draft_k, dim=-1) 

            if vocab_map is not None:
                draft_next_token = vocab_map[draft_next_token]
                
            draft_confidences=draft_confidences.unsqueeze(-1)*next_token_values 

            draft_top_k_token_values, draft_top_k_token_indices=torch.topk(draft_confidences.view(bsz, -1), k=draft_k, dim=-1) 
            
            past_kv_len=draft_past_key_values_tree[0][0].shape[-2]
            
            for idx_tree in range(len(trees)):
                for idx_seq in range(draft_k):
                    for idx_k in range(draft_k):
                        
                        index=draft_k + draft_k*draft_k*(idx_token-1) + idx_seq*draft_k + idx_k
                            
                        node=Node(depth=idx_token, input_id=None)

                        assert trees[idx_tree][index]==0
                        trees[idx_tree][index]=node
                        

            total_input_ids.append(draft_next_token.view(bsz, -1)) 
            confidences.append(draft_confidences.view(bsz, -1)) 
            
            draft_next_token=draft_next_token.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            draft_confidences=draft_confidences.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            
            
            draft_top_k_token_indices_div_k=draft_top_k_token_indices//draft_k
            
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, next_feature_states.shape[-1])
            next_feature_states=next_feature_states.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, idx_token)
            attention_seen_indices=attention_seen_indices.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            
            cur_attention_seen_indices=torch.arange(past_kv_len, past_kv_len+draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1) 
            attention_seen_indices=torch.concat([attention_seen_indices, cur_attention_seen_indices], dim=-1)
            
            parents_list.append(draft_top_k_token_indices)

        parents_list=torch.stack(parents_list, dim=1) 
        parents_list=parents_list.to(torch.int8).cpu().tolist()
                
        total_input_ids=torch.concat(total_input_ids, dim=-1)
        total_position_ids=torch.concat(total_position_ids, dim=1)
        confidences=torch.concat(confidences, dim=-1)
        
        chosen_index=torch.topk(confidences, k=draft_total_token, dim=-1)
        chosen_index, _=torch.sort(chosen_index.indices, dim=-1, descending=False)
        chosen_index_list=chosen_index.to(torch.int16).cpu().tolist()
        
        for idx_tree, tree in enumerate(trees):
            for index in chosen_index_list[idx_tree]:
                node=tree[index]
                
                node.child=[]
                node.input_id=total_input_ids[idx_tree][index]
                
                if index < draft_k:
                    node.parent=[]
                    
                elif index < (draft_k+draft_k*draft_k):
                    parent_index=(index-draft_k)//draft_k
                    
                    node.parent=[parent_index]
                    tree[parent_index].child.append(index)
                    
                else:
                    
                    parent_index1=(index-draft_k) // (draft_k*draft_k)
                    parent_index2=(index-draft_k - parent_index1*draft_k*draft_k)//draft_k
                    parent_index2=parents_list[idx_tree][parent_index1-1][parent_index2]
                    
                    parent_index=draft_k + (parent_index1-1)*draft_k*draft_k + parent_index2

                    node.parent=tree[parent_index].parent+[parent_index]
                    tree[parent_index].child.append(index)
                    
                    
        next_token_trees=total_input_ids.gather(index=chosen_index, dim=-1)
        target_position_ids=total_position_ids.gather(index=chosen_index, dim=-1)
        return {
            'trees':trees,
            'trees_chosen_index':chosen_index_list,
            'next_token_trees':next_token_trees,
            'target_position_ids':target_position_ids
        }


    def get_attention_mask(past_seq_len,q_length,dtype,bsz=1,device='cuda',padding_positions=None):

        min_dtype = torch.finfo(dtype).min
        
        kv_length=past_seq_len+q_length
        
        attention_mask=torch.triu(torch.full((q_length,kv_length),fill_value=min_dtype,dtype=dtype,device=device)
                            , diagonal=kv_length-q_length+1)
        attention_mask=attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz,1,1,1)
        
        if isinstance(padding_positions, torch.Tensor):
            padding_positions_tensor=padding_positions
        
            attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype

        elif padding_positions:
            
            batch_indices = []
            pos_indices = []

            for batch_id, pad_positions in enumerate(padding_positions):
                for pos in pad_positions:
                    batch_indices.append(batch_id)
                    pos_indices.append(pos)

            if batch_indices: 

                attention_mask[batch_indices, 0, :, pos_indices] = min_dtype
            

        return attention_mask


    start_time=time.time()
    
    input_ids=inputs.input_ids
    attention_mask=inputs.attention_mask
    target_past_key_values=DynamicCache()
    
    avg_acc_length=[0,0]
    eos_token_id=model.target_model.config.eos_token_id
    bsz=input_ids.shape[0]
    end_sig=[0]*bsz
    device=model.target_model.device
    
    transfer_stream = torch.cuda.Stream(device)
    executor = ThreadPoolExecutor(max_workers=64)

    prefill_time_start=time.time()
    target_time_start=time.time()
    
    global total_target_time, total_draft_time, total_check_time
    
    total_target_time, total_draft_time, total_check_time = 0, 0, 0

    all_draft_input_states=None
    all_draft_input_ids=None

    position_ids=[torch.sum(item) for item in attention_mask]
    past_position_ids=[item.item()-1 for item in position_ids]

    position_ids=[torch.concat(
        [torch.zeros((input_ids.shape[-1]-item)),torch.arange(0,item)],dim=-1)
                for item in position_ids] 
    position_ids=torch.stack(position_ids,dim=0)

    padding_positions=[]
    for example in attention_mask:
        cur_padding_positions=set()

        for idx, cur_attention_mask in enumerate(example):
            if not cur_attention_mask:
                cur_padding_positions.add(idx)

        padding_positions.append(cur_padding_positions)

    input_ids=input_ids.to(device)
    attention_mask=get_attention_mask(0,attention_mask.shape[-1],model.target_model.dtype,
                                        bsz,device=device,padding_positions=padding_positions)
    position_ids=position_ids.to(device)

    with torch.amp.autocast(str(model.target_model.device),
                                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
        target_outputs=model.target_model(**inputs, position_ids=position_ids,
                                    past_key_values=target_past_key_values,use_cache=True,
                                    output_hidden_states=True,logits_to_keep=1)

        feature_states=target_outputs.hidden_states[-1]
        target_logits=target_outputs.logits[:, -1:, :]
    

    if statistical_time:
        torch.cuda.synchronize()
        total_target_time+=time.time()-target_time_start
            
    if do_sample==False:
        target_next_token=target_logits.softmax(dim=-1).argmax(-1)
    elif do_sample==True:
        target_next_token=sampling(target_logits,top_k,top_p,temperature,eos_token_id)
    else:
        raise ValueError('"do_sample" must be True or False')
    
    generated_sequences=target_next_token
    
    draft_input_ids=torch.concat([input_ids[:,1:],target_next_token],dim=-1)
    draft_attention_mask=get_attention_mask(0, attention_mask.shape[-1]-1, model.dtype,
                                        bsz, device=device, padding_positions=padding_positions)
        
    if return_all_draft_input:
        all_draft_input_states=feature_states
        all_draft_input_ids=draft_input_ids

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start=time.time()

    with torch.amp.autocast(str(model.target_model.device),
                                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
        draft_outputs=model(hidden_states=feature_states[:, :-1, :].to(model.dtype), router=1, input_ids=None,
                            attention_mask=draft_attention_mask, position_ids=position_ids[:, :-1], use_cache=True)
        
        draft_past_key_values=draft_outputs['past_key_values']
        
        draft_attention_mask=get_attention_mask(attention_mask.shape[-1]-1, 1, model.dtype,
                                        bsz, device=device, padding_positions=padding_positions)
        
        draft_outputs=model(hidden_states=feature_states[:, -1:, :].to(model.dtype), router=2, input_ids=target_next_token,
                            attention_mask=draft_attention_mask, position_ids=position_ids[:, -1:], use_cache=True,
                            past_key_values=draft_past_key_values)
    
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time+=time.time()-draft_time_start

    draft_past_key_values=draft_outputs['past_key_values']
    draft_hidden_states=draft_outputs['hidden_states'][:,-1:,:]
    next_feature_states=draft_outputs['next_feature_states'][:,-1:,:]
    
    if repeated_generate_nums is not None and repeated_generate_nums>1:
        
        target_next_token=target_next_token.repeat_interleave(repeated_generate_nums,dim=0)
        target_past_key_values.batch_repeat_interleave(repeated_generate_nums)
        
        new_past_key_values=[]
        for cur_past_key_values in draft_past_key_values:
            cur_past_key_values=[cur_past_key_values[0].repeat_interleave(repeated_generate_nums,dim=0),
                                cur_past_key_values[1].repeat_interleave(repeated_generate_nums,dim=0)]
            new_past_key_values.append(cur_past_key_values)
        draft_past_key_values=new_past_key_values
        draft_hidden_states=draft_hidden_states.repeat_interleave(repeated_generate_nums,dim=0)
        next_feature_states=next_feature_states.repeat_interleave(repeated_generate_nums,dim=0)

        generated_sequences=generated_sequences.repeat_interleave(repeated_generate_nums,dim=0)
        bsz*=repeated_generate_nums
        end_sig=[0]*bsz
        
        new_past_position_ids=[]
        for cur_past_position_ids in past_position_ids:
            for _ in range(repeated_generate_nums):
                new_past_position_ids.append(cur_past_position_ids)
        past_position_ids=new_past_position_ids
        
        new_padding_positions=[]
        for cur_padding_positions in padding_positions:
            for _ in range(repeated_generate_nums):
                new_padding_positions.append(deepcopy(cur_padding_positions))
        padding_positions=new_padding_positions
        
        if return_all_draft_input:
            all_draft_input_states=all_draft_input_states.repeat_interleave(repeated_generate_nums,dim=0)
            all_draft_input_ids=all_draft_input_ids.repeat_interleave(repeated_generate_nums,dim=0)
            
    padding_positions_indices=[]
        
    for batch_id, pad_positions in enumerate(padding_positions):
        for pos in pad_positions:
            padding_positions_indices.append([batch_id, pos])

    if padding_positions_indices:  

        padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
        
    padding_positions_tensor=padding_positions_indices
    
    past_position_ids_tensor=torch.tensor(past_position_ids, dtype=torch.int16).to(device).long() 
    
    draft_input_states_dict={}
    draft_input_ids_dict={}
    generated_sequences_dict={}
    padding_positions_dict={}
    residual_index=[_ for _ in range(bsz)]
        

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start=time.time()

    with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
    
        
        outputs=draft_generate(model,next_feature_states,draft_hidden_states,draft_past_key_values,
                            draft_token_length,past_position_ids_tensor,padding_positions_tensor,
                            draft_k=draft_k, draft_total_token=draft_total_token,vocab_map=vocab_map)
        
        draft_trees=outputs['trees']
        trees_chosen_index=outputs['trees_chosen_index']
        next_token_trees=outputs['next_token_trees']
        target_position_ids=outputs['target_position_ids']

    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time+=time.time()-draft_time_start

    total_prefill_time=time.time()-prefill_time_start

    for token_num in range(1,max_new_token):

        past_kv_len=target_past_key_values.get_seq_length()
        kv_length=past_kv_len+draft_total_token+1 
        q_length=draft_total_token+1

        target_trees=draft_trees
        
        for idx_tree, tree in enumerate(target_trees):
            unique_index=past_kv_len+1 
            
            for index in trees_chosen_index[idx_tree]:
                
                node=tree[index]
                node.target_index=unique_index
                unique_index+=1
                
        next_token_trees=torch.concat([target_next_token, next_token_trees], dim=-1) 
        target_position_ids = target_position_ids+2
        target_position_ids = torch.concat([(past_position_ids_tensor+1).unsqueeze(-1), target_position_ids], dim=-1) 

        min_dtype=torch.finfo(model.target_model.dtype).min
        
        target_attention_mask=torch.zeros((q_length, kv_length), dtype=model.target_model.dtype,device=device)
        target_attention_mask[..., past_kv_len+1:]=min_dtype 
        target_attention_mask=target_attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz,1,1,1)
        
        indices=[]
        
        for idx_tree, tree in enumerate(target_trees):
            
            for index in trees_chosen_index[idx_tree]:
                
                node=tree[index]
                
                cur_index=node.target_index-past_kv_len
                for index in node.parent:
                    
                    seen_token_index=tree[index].target_index
                    indices.append([idx_tree, cur_index, seen_token_index])

                indices.append([idx_tree, cur_index, node.target_index])
                        
        if indices:
            indices=torch.tensor(indices, dtype=torch.int16).to(device).long()
            target_attention_mask[indices[:, 0], 0, indices[:, 1], indices[:, 2]] = 0
            

        if isinstance(padding_positions_tensor, torch.Tensor):
            target_attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype
            
        def transfer_input_ids(trees, trees_chosen_index, device, stream):
            torch.cuda.set_device(device)
            
            with torch.cuda.stream(stream):
                for idx_tree, tree in enumerate(trees):
                    for index in trees_chosen_index[idx_tree]:
                        
                        node=tree[index]
                        node.input_id=node.input_id.cpu()
                        
            return trees
        
        transfer_thread = executor.submit(transfer_input_ids, target_trees, trees_chosen_index, device, transfer_stream)
        
        if statistical_time:
            torch.cuda.synchronize()
            target_time_start=time.time()

        with torch.amp.autocast(str(model.target_model.device),
                            dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
            
            target_outputs=model.target_model.model.language_model(input_ids=next_token_trees, attention_mask=target_attention_mask,
                                            past_key_values=target_past_key_values, position_ids=target_position_ids,
                                            use_cache=True, output_hidden_states=True)
            
            feature_states_tree=target_outputs.hidden_states[-1]
            target_outputs_logits=model.target_model.lm_head(feature_states_tree)
            
            if do_sample==False:
                target_next_token_tree=target_outputs_logits.softmax(-1).argmax(-1)
            elif do_sample==True:
                target_next_token_tree=sampling(target_outputs_logits,top_k,top_p,temperature,eos_token_id)
            else:
                raise ValueError('"do_sample" must be True or False')
        
        if statistical_time:
            torch.cuda.synchronize()
            total_target_time+=time.time()-target_time_start

        target_trees=transfer_thread.result()

        acc_length=[0]*bsz
        chosen_index=[[] for _ in range(bsz)]
        next_token=[[] for _ in range(bsz)]
            
        target_next_token_tree_list=target_next_token_tree.tolist()
        
        for idx_tree, tree in enumerate(target_trees):
            
            if end_sig[idx_tree]==0:
                cur_next_token=[target_next_token_tree_list[idx_tree][0]]
                cur_acc_length=1
                cur_chosen_index=[past_kv_len]
                tmp_sig=True
                
                for index in trees_chosen_index[idx_tree]:
                    
                    if index >= draft_k:
                        break
                
                    node=tree[index]
                    
                    if cur_next_token[-1]==node.input_id.item(): 
                        cur_next_token.append(target_next_token_tree_list[idx_tree][node.target_index-past_kv_len])
                        cur_acc_length+=1
                        cur_chosen_index.append(node.target_index)
                        
                        while tmp_sig:
                            tmp_sig=False
                            
                            for child_index in node.child:
                                child=tree[child_index]
                                
                                if cur_next_token[-1]==child.input_id.item(): 
                                    cur_next_token.append(target_next_token_tree_list[idx_tree][child.target_index-past_kv_len])
                                    cur_acc_length+=1
                                    cur_chosen_index.append(child.target_index)
                                    
                                    tmp_sig=True 
                                    node=child
                                    break
                                    
                        break

                acc_length[idx_tree]=cur_acc_length
                next_token[idx_tree]=cur_next_token
                chosen_index[idx_tree]=cur_chosen_index
                
                for cur_idx, token in enumerate(next_token[idx_tree]):

                    if token==eos_token_id:
                        end_sig[idx_tree]=1
                        
                        chosen_index[idx_tree]=chosen_index[idx_tree][:cur_idx+1]
                        next_token[idx_tree]=next_token[idx_tree][:cur_idx+1]
                        acc_length[idx_tree]=cur_idx+1
                        break
                
                avg_acc_length[0]=avg_acc_length[0]+acc_length[idx_tree]
                avg_acc_length[1]+=1
                        
            else:
                acc_length[idx_tree]=0
        max_acc_length=max(acc_length)
        last_valid_index=[max_acc_length-1]*bsz 

        for idx_batch in range(bsz):

            cur_index_length=len(chosen_index[idx_batch])
            if cur_index_length<max_acc_length:
                padding_num=max_acc_length-cur_index_length
                new_chosen_index=[]
                new_next_token=[]

                cur_index=past_kv_len 
                for index,token in zip(chosen_index[idx_batch],next_token[idx_batch]):
                    if padding_num>0:
                        
                        if index!=cur_index:
                            while cur_index!=index and padding_num!=0:
                                new_chosen_index.append(cur_index)
                                new_next_token.append(eos_token_id)
                                padding_positions[idx_batch].add(cur_index) 
                                cur_index+=1
                                padding_num-=1

                            new_chosen_index.append(index)
                            new_next_token.append(token)
                            last_valid_index[idx_batch]=len(new_chosen_index)-1
                            cur_index+=1
                            
                        else:
                            new_chosen_index.append(index)
                            new_next_token.append(token)
                            last_valid_index[idx_batch]=len(new_chosen_index)-1
                            cur_index+=1

                    else:
                        new_chosen_index.append(index) 
                        new_next_token.append(token)
                        last_valid_index[idx_batch]=len(new_chosen_index)-1
                        
                while padding_num>0:
                    new_chosen_index.append(cur_index)
                    new_next_token.append(eos_token_id)
                    
                    padding_positions[idx_batch].add(cur_index)
                    cur_index+=1
                    padding_num-=1

                chosen_index[idx_batch]=new_chosen_index
                next_token[idx_batch]=new_next_token

        feature_states_index=[[] for _ in range(bsz)]

        for idx_batch in range(bsz):
            for index in chosen_index[idx_batch]:
                
                feature_states_index[idx_batch].append(index-past_kv_len)
                
                
                    
        next_token=torch.tensor(next_token, device=device)
        last_valid_index=torch.tensor(last_valid_index, dtype=torch.int16).unsqueeze(-1).to(device).long() 
        target_next_token=next_token.gather(index=last_valid_index, dim=-1)
        feature_states_index=torch.tensor(feature_states_index, dtype=torch.int8).to(device).long()
        

        
        B,T,D=feature_states_tree.shape
        feature_states_index=feature_states_index.unsqueeze(-1).expand(B,-1,D)
        feature_states=feature_states_tree.gather(dim=1, index=feature_states_index)
        
        if return_all_draft_input:
            all_draft_input_states=torch.concat([all_draft_input_states, feature_states], dim=1)
            all_draft_input_ids=torch.concat([all_draft_input_ids, next_token], dim=-1)
            
        generated_sequences=torch.concat([generated_sequences,next_token],dim=-1)


        
        if 0 not in end_sig:
            break
        if generated_sequences.shape[-1]>=max_new_token:
            break
        

        idx_batch=0
        while idx_batch < bsz:
            
            if end_sig[idx_batch]==1:

                delete_idx=idx_batch
                ori_idx=residual_index[idx_batch]
                end_sig=[end_sig[_] for _ in range(bsz) if _ != delete_idx]

                chosen_index=[chosen_index[_] for _ in range(bsz) if _ != delete_idx]
                
                for kv_idx in range(len(target_past_key_values.key_cache)):
                    key = target_past_key_values.key_cache[kv_idx]        
                    value = target_past_key_values.value_cache[kv_idx]
                    
                    target_past_key_values.key_cache[kv_idx]=torch.concat(
                        [key[:delete_idx], key[delete_idx+1:]], dim=0)
                    target_past_key_values.value_cache[kv_idx]=torch.concat(
                        [value[:delete_idx], value[delete_idx+1:]], dim=0)
                
                new_past_key_values=[]
                for cur_past_key_values in draft_past_key_values:
                    
                    draft_key=torch.concat(
                        [cur_past_key_values[0][:delete_idx], cur_past_key_values[0][delete_idx+1:]], dim=0)
                    draft_value=torch.concat(
                        [cur_past_key_values[1][:delete_idx], cur_past_key_values[1][delete_idx+1:]], dim=0)
                    new_past_key_values.append([draft_key, draft_value])
                    
                draft_past_key_values=new_past_key_values
                        
                next_token=torch.concat([next_token[:delete_idx], next_token[delete_idx+1:]], dim=0)
                target_next_token=torch.concat([target_next_token[:delete_idx], target_next_token[delete_idx+1:]], dim=0)
                
                feature_states=torch.concat([feature_states[:delete_idx], feature_states[delete_idx+1:]], dim=0)
                last_valid_index=torch.concat([last_valid_index[:delete_idx], last_valid_index[delete_idx+1:]], dim=0)
                
                padding_positions_dict[str(ori_idx)]=padding_positions[delete_idx]
                padding_positions=[padding_positions[_] for _ in range(bsz) if _ != delete_idx]
                
                past_position_ids=[item for idx_tmp, item in enumerate(past_position_ids) if idx_tmp != delete_idx]

                generated_sequences_dict[str(ori_idx)]=generated_sequences[delete_idx]
                generated_sequences=torch.concat(
                    [generated_sequences[:delete_idx], generated_sequences[delete_idx+1:]], dim=0)
                
                if return_all_draft_input:
                    draft_input_states_dict[str(ori_idx)]=all_draft_input_states[delete_idx]
                    all_draft_input_states=torch.concat(
                        [all_draft_input_states[:delete_idx], all_draft_input_states[delete_idx+1:]], dim=0)
                    
                    draft_input_ids_dict[str(ori_idx)]=all_draft_input_ids[delete_idx]
                    all_draft_input_ids=torch.concat(
                        [all_draft_input_ids[:delete_idx], all_draft_input_ids[delete_idx+1:]], dim=0)
                    
                residual_index=[residual_index[_] for _ in range(bsz) if _ != delete_idx]
                bsz-=1
                
            else:   
                idx_batch+=1
                
                
                
        padding_positions_indices=[]
        
        for batch_id, pad_positions in enumerate(padding_positions):
            for pos in pad_positions:
                padding_positions_indices.append([batch_id, pos])

        if padding_positions_indices: 

            padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
            
        padding_positions_tensor=padding_positions_indices

        
        
        for idx_batch in range(bsz):
            chosen_index[idx_batch]=[x for x in range(past_kv_len)]+chosen_index[idx_batch] 

        prefix_length=target_past_key_values.get_seq_length()

        for idx_batch in range(len(chosen_index)):
            the_prefix_length=0
            
            for idx,index in enumerate(chosen_index[idx_batch]):
                if idx==index:
                    the_prefix_length=idx+1
                else:
                    break
            prefix_length=min(the_prefix_length,prefix_length)
            
        if prefix_length==len(chosen_index[0]):
            target_past_key_values.crop(prefix_length)
            
        else:
            
            chosen_index=[[x-prefix_length for x in item[prefix_length:]] for item in chosen_index]
            chosen_index=torch.tensor(chosen_index,device=device)
            
            target_past_key_tensor=torch.stack(target_past_key_values.key_cache, dim=0)
            target_past_value_tensor=torch.stack(target_past_key_values.value_cache, dim=0)
            
            L, B, H, T, D = target_past_key_tensor.shape
            index_expanded = chosen_index.unsqueeze(1).unsqueeze(-1).unsqueeze(0) 
            index_expanded = index_expanded.expand(L, B, H, -1, D)      
            
            prefix_key=target_past_key_tensor[..., :prefix_length,:]
            prefix_value=target_past_value_tensor[..., :prefix_length,:]
            
            suffix_key=target_past_key_tensor[..., prefix_length:,:]
            suffix_value=target_past_value_tensor[..., prefix_length:,:]

            suffix_key = suffix_key.gather(dim=-2, index=index_expanded)         
            suffix_value = suffix_value.gather(dim=-2, index=index_expanded)    
            
            new_key=torch.concat([prefix_key,suffix_key],dim=-2)
            new_value=torch.concat([prefix_value,suffix_value],dim=-2)
        
            for idx_L in range(L):

                target_past_key_values.key_cache[idx_L] = new_key[idx_L]
                target_past_key_values.value_cache[idx_L] = new_value[idx_L]
                
        
        draft_attention_mask=get_attention_mask(draft_past_key_values[0][0].shape[-2], max_acc_length,
                                                model.dtype, bsz, padding_positions=padding_positions_tensor)
        draft_position_ids=[[] for _ in range(bsz)]
        
        
        assert target_past_key_values.get_seq_length()==draft_past_key_values[0][0].shape[-2]+max_acc_length
        
        draft_position_ids=[[] for _ in range(bsz)]
        
        for idx_batch in range(bsz):
            cur_position_ids=past_position_ids[idx_batch]
            cur_index=draft_past_key_values[0][0].shape[-2]
            
            for idx_token in range(max_acc_length):
                if cur_index not in padding_positions[idx_batch]:
                    cur_position_ids+=1
                    
                draft_position_ids[idx_batch].append(cur_position_ids)
                cur_index+=1
                
            past_position_ids[idx_batch]=draft_position_ids[idx_batch][-1]
            
            
        draft_position_ids=torch.tensor(draft_position_ids, dtype=torch.int16).to(device).long()
        past_position_ids_tensor=draft_position_ids[:, -1]

            
        if statistical_time:
            torch.cuda.synchronize()
            draft_time_start=time.time()

        with torch.amp.autocast(str(model.target_model.device),
                            dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
            if statistical_time:
                torch.cuda.synchronize()
                check_time_start=time.time()

            draft_outputs=model(hidden_states=feature_states.to(model.dtype), router=2, input_ids=next_token,
                        attention_mask=draft_attention_mask,use_cache=True,
                        position_ids=draft_position_ids,past_key_values=draft_past_key_values)
            
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time+=time.time()-check_time_start 

            draft_past_key_values=draft_outputs['past_key_values']
            
            B, S, D =draft_outputs['hidden_states'].shape
            last_valid_index=last_valid_index.unsqueeze(-1).expand(-1,-1,D) 
            draft_hidden_states=draft_outputs['hidden_states'].gather(index=last_valid_index, dim=1)
            next_feature_states=draft_outputs['next_feature_states'].gather(index=last_valid_index, dim=1)

            outputs=draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values,
                                draft_token_length, past_position_ids_tensor, padding_positions_tensor,
                                draft_k=draft_k, draft_total_token=draft_total_token, vocab_map=vocab_map)
            
            draft_trees=outputs['trees']
            trees_chosen_index=outputs['trees_chosen_index']
            next_token_trees=outputs['next_token_trees']
            target_position_ids=outputs['target_position_ids']

        if statistical_time:
            torch.cuda.synchronize()
            total_draft_time+=time.time()-draft_time_start
            
            
        
    post_time_start=time.time()
            
    for idx_batch in range(bsz):
        delete_idx=idx_batch
        ori_idx=residual_index[idx_batch] 
        
        padding_positions_dict[str(ori_idx)]=padding_positions[delete_idx]
        generated_sequences_dict[str(ori_idx)]=generated_sequences[delete_idx]
        
        if return_all_draft_input:
            draft_input_states_dict[str(ori_idx)]=all_draft_input_states[delete_idx]
            draft_input_ids_dict[str(ori_idx)]=all_draft_input_ids[delete_idx]
        
    bsz=len(generated_sequences_dict)

    if return_all_draft_input:

        all_draft_input_states_without_padding=[]
        all_draft_input_ids_without_padding=[]
        
        for idx_batch in range(bsz):
            chosen_index=[]
            cur_draft_input_states=draft_input_states_dict[str(idx_batch)]
            cur_draft_input_ids=draft_input_ids_dict[str(idx_batch)]
            
            for index in range(cur_draft_input_ids.shape[-1]):
                if index not in padding_positions_dict[str(idx_batch)]:
                    chosen_index.append(index)
                    
            all_draft_input_states_without_padding.append(cur_draft_input_states[chosen_index,:])
            all_draft_input_ids_without_padding.append(cur_draft_input_ids[chosen_index])
            
        all_draft_input_states=all_draft_input_states_without_padding
        all_draft_input_ids=all_draft_input_ids_without_padding
            
    new_padding_positions=[[] for _ in range(bsz)]
    for idx_batch in range(bsz):
        cur_padding_positions=[]
        
        for index in sorted(padding_positions_dict[str(idx_batch)]):

            if (index+1)>=input_ids.shape[-1]: 
                cur_padding_positions.append(index-input_ids.shape[-1]+1) 
                
        new_padding_positions[idx_batch]=cur_padding_positions
    

    generated_sequences=generated_sequences.tolist()
    
    
    filtered_generated_token_ids=[]
    max_sequence_length=0

    
    for idx_batch in range(bsz):
        generated_sequence=generated_sequences_dict[str(idx_batch)]
        cur_position_ids=new_padding_positions[idx_batch]
        
        sequence_without_padding=[]
        cur_padding_index=0
        
        for idx_token, token in enumerate(generated_sequence):
            
            if cur_padding_index<len(cur_position_ids):
                if idx_token==cur_position_ids[cur_padding_index]:
                    cur_padding_index+=1
                    continue
                else:
                    sequence_without_padding.append(token)
            else:
                sequence_without_padding.append(token)

            if token==eos_token_id:
                break
            
        filtered_generated_token_ids.append(sequence_without_padding)
        max_sequence_length=max(max_sequence_length,len(sequence_without_padding))
        
    return {
        'generated_token_ids':filtered_generated_token_ids,
        'max_sequence_length':max_sequence_length,
        'total_acc_length':avg_acc_length[0],
        'total_decoded_token_num':avg_acc_length[1],
        'total_time_cost':time.time()-start_time,
        'target_time_cost':total_target_time,
        'draft_time_cost':total_draft_time,
        'check_time_cost':total_check_time,
        'prefill_time_cost':total_prefill_time,
        'post_time_cost':time.time()-post_time_start,
        'all_draft_input_states':all_draft_input_states,
        'all_draft_input_ids':all_draft_input_ids
    }  
    
    
