import math
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import time
import sys

from transformers.activations import ACT2FN
from transformers import AutoTokenizer
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
import copy

     
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):

    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min,dtype=dtype, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):

    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

    if hidden_states.dim()==4:
        
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states.unsqueeze(-3).expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
    elif hidden_states.dim()==5:
    
        forward_num, batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states.unsqueeze(-3).expand(forward_num, batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(forward_num, batch, num_key_value_heads * n_rep, slen, head_dim)
    
    else:
        raise ValueError()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  
    sin = sin.squeeze(1).squeeze(0) 
    cos = cos[position_ids].unsqueeze(1)  
    sin = sin[position_ids].unsqueeze(1)  
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DraftRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class DraftLinearScalingRotaryEmbedding(DraftRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class DraftnamicNTKScalingRotaryEmbedding(DraftRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)




class DraftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=config.torch_dtype)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class DraftDecodeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=config.torch_dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=config.torch_dtype)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, inputs_embeds):

        down_proj = self.down_proj(self.act_fn(self.gate_proj(inputs_embeds)) * self.up_proj(hidden_states))

        return down_proj


class DraftRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)
    
    
    
class DraftAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=config.torch_dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=config.torch_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=config.torch_dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=config.torch_dtype)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DraftRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DraftLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DraftnamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            use_align: Optional[bool] = False,
            forward_num: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if use_align and forward_num>1:
            _, bsz, q_len, hidden_size = hidden_states.size() 
            
            query_states = self.q_proj(hidden_states[-1])
            all_hidden_states = hidden_states

            key_states = self.k_proj(all_hidden_states)
            value_states = self.v_proj(all_hidden_states)
            
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(forward_num, bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(2, 3)
            value_states = value_states.view(forward_num, bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(2, 3)

            kv_seq_len = key_states.shape[-2]
            past_kv_len=past_key_value[0].shape[-2]
            kv_seq_len += past_key_value[0].shape[-2]
                
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            
            key_states = torch.cat([past_key_value[0][None, ...].expand(forward_num, -1, -1, -1, -1), key_states], dim=3)
            value_states = torch.cat([past_key_value[1][None, ...].expand(forward_num, -1, -1, -1, -1), value_states], dim=3)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states[None, ...], key_states.transpose(3, 4)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (forward_num, bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(forward_num, bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
                
            align_masks = None
            for idx in range(forward_num):

                if idx == 0:
                    align_small_mask = torch.tril(torch.ones((q_len - forward_num + idx + 1, q_len - forward_num + idx + 1),
                                                            device=attn_weights.device, dtype=query_states.dtype))
                else:
                    align_small_mask = torch.eye(q_len - forward_num + idx + 1, device=attn_weights.device, dtype=query_states.dtype)
                    
                align_big_mask = torch.zeros((q_len, q_len), device=attn_weights.device,dtype=query_states.dtype)
                align_big_mask[forward_num - idx - 1:, :q_len - forward_num + idx + 1] = align_small_mask
                
                if idx == 0:
                    align_big_mask=torch.concat([torch.ones((q_len, past_kv_len), device=attn_weights.device, dtype=query_states.dtype),
                                                align_big_mask], dim=-1)
                else:
                    align_big_mask=torch.concat([torch.zeros((q_len, past_kv_len), device=attn_weights.device,dtype=query_states.dtype),
                                                align_big_mask], dim=-1)
                
                align_big_mask = align_big_mask.view(1, bsz, 1, q_len, kv_seq_len)

                if align_masks is None:
                    align_masks = align_big_mask
                else:
                    align_masks = torch.cat([align_masks, align_big_mask], dim=0)  
                
            total_attn_weights = (attn_weights * align_masks).sum(dim=0)
        
            total_attn_weights = nn.functional.softmax(total_attn_weights + attention_mask, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights = total_attn_weights[None, ...] * align_masks

            attn_output = torch.matmul(attn_weights, value_states).sum(dim=0) 

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads*self.head_dim)

            attn_output = self.o_proj(attn_output)    
            
            return attn_output, None
            
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads*self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class DraftDecoderLayeremb(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = DraftAttention(config=config)
        self.mlp = DraftMLP(config)
        self.input_layernorm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: Optional[bool] = False,
            use_align: Optional[bool] = False,
            forward_num: Optional[int] = 0,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if use_align and forward_num>1:
            residual = hidden_states[-1]
            hidden_states = self.input_layernorm(hidden_states) 
            
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)


        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            use_align=use_align, 
            forward_num=forward_num, 
        )
        hidden_states = residual + hidden_states


        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = [hidden_states,present_key_value]

        return outputs



class DraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dtype=config.torch_dtype
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.prefill_mlp=DraftMLP(config)
        self.decode_mlp=DraftDecodeMLP(config)
        self.arlayer = DraftDecoderLayeremb(config)

        self.norm = DraftRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def init_draft_decode_router(self):
        self.draft_decode_mlp=copy.deepcopy(self.decode_mlp)
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:

            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def forward(
            self,
            hidden_states,
            inputs_embeds,
            router,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List] = None,
            use_cache: Optional[bool] = None,
            use_align: Optional[bool] = False,
            forward_num: Optional[int] = 0,
            past_hidden_states = None,
    ):
        assert router==1 or router==2 or router==3

        bsz, seq_len, _ = hidden_states.shape
        seq_length_with_past = seq_len
        past_key_values_length = 0


        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (bsz, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        
        if attention_mask.dim() != 4:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (bsz, seq_len), hidden_states, past_key_values_length
            )

        key_value_cache = []
        
        if router==1:
            hidden_states = hidden_states.to(self.dtype)
            
            residual = hidden_states
            hidden_states=self.prefill_mlp(hidden_states)
            hidden_states=hidden_states+residual
            
            residual = hidden_states
            hidden_states = self.arlayer.input_layernorm(hidden_states)
            
            bsz, q_len, _ = hidden_states.size()

            key_states = self.arlayer.self_attn.k_proj(hidden_states)
            value_states = self.arlayer.self_attn.v_proj(hidden_states)
            query_states = torch.zeros((bsz, self.arlayer.self_attn.num_heads, q_len, self.arlayer.self_attn.head_dim),device=key_states.device)

            key_states = key_states.view(bsz, q_len, self.arlayer.self_attn.num_key_value_heads, self.arlayer.self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.arlayer.self_attn.num_key_value_heads, self.arlayer.self_attn.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]

            cos, sin = self.arlayer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            past_key_value = [key_states, value_states]
            key_value_cache += [past_key_value]
            
            return {
                'hidden_states':None,
                'past_key_values':key_value_cache,
                'next_feature_states':None
            }
            

        if use_align and forward_num>1:
            hidden_states = hidden_states.to(self.dtype)
            inputs_embeds = inputs_embeds.to(self.dtype)
            
            hidden_states=torch.concat([hidden_states.unsqueeze(0), past_hidden_states], dim=0)
            residual = hidden_states
            
            hidden_states=self.draft_decode_mlp(hidden_states, inputs_embeds.unsqueeze(0).expand(hidden_states.shape[0], -1, -1, -1))
            hidden_states=hidden_states+residual
            
        elif router==2:
            
            hidden_states = hidden_states.to(self.dtype)
            inputs_embeds = inputs_embeds.to(self.dtype)

            residual = hidden_states
            hidden_states=self.decode_mlp(hidden_states,inputs_embeds)
            hidden_states=hidden_states+residual
            
        elif router==3:
            hidden_states = hidden_states.to(self.dtype)
            inputs_embeds = inputs_embeds.to(self.dtype)

            residual = hidden_states
            hidden_states=self.draft_decode_mlp(hidden_states,inputs_embeds)
            hidden_states=hidden_states+residual
                
        
        idx=0
        decoder_layer=self.arlayer

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            use_align=use_align, 
            forward_num=forward_num, 
        )

        hidden_states = layer_outputs[0]
        if use_cache:
            key_value_cache += [layer_outputs[1]]

        hidden_states=self.norm(hidden_states)
     
        return {
            'hidden_states':hidden_states,
            'past_key_values':key_value_cache,
            'next_feature_states':hidden_states
        }


class Model(nn.Module):
    def __init__(self, config, target_model, path=None):
        super().__init__()

        self.dtype=config.torch_dtype
        self.target_model = target_model
        self.draft_model = DraftModel(config)
        self.device=target_model.device
        
        self.embed_tokens = self.target_model.model.language_model.embed_tokens
        self.lm_head = self.target_model.lm_head

        for param in self.target_model.parameters():
            param.requires_grad = False
            
        if path is not None:
            self.load_model(path)


    def load_model(self, load_path):

        saved_state_dict = torch.load(load_path)

        self.draft_model.load_state_dict(saved_state_dict['draft_model'])

            
    def save_model(self, save_path):
        
        state_dict = {
            'draft_model': self.draft_model.state_dict()
        }

        torch.save(state_dict, save_path)

    def forward(
            self,
            hidden_states,
            input_ids,
            router,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List] = None,
            use_cache: Optional[bool] = None,
            use_align: Optional[bool] = False,
            forward_num: Optional[int] = 0,
            past_hidden_states = None,
    ):
        if router==1:
            inputs_embeds=None
        else:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)

        outputs=self.draft_model(hidden_states,inputs_embeds,router,attention_mask,
                                position_ids,past_key_values,use_cache,
                                use_align, forward_num, past_hidden_states)
     
        return outputs