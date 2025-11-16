## PASC: Bridging the Training-Testing Gap in VLM Speculative Decoding with Phase-Aware Routing and Self-Conditioning

To train a draft model, you should run `train_draft.py` first. Take target model as Qwen2.5-VL-7B for example:

```bash
python train_draft.py \
  --model_dir path/to/qwen2.5vl \
  --adapter_path path/to/save/adapter \
  --version_name qwen2_5vl_reproduce_draft \
  --num_epochs 3 \
  --lr 5e-5 \
  --accumulation_steps 16 \
  --warmup_ratio 0.05 \
  --sample_num 100 \
  --batch_size 1 \
  --num_workers 1 \
  --coco_path path/to/coco2017 \
  --sharegpt4v_path path/to/sharegpt4vcaption
```

Then train the draft model in the second phase:

```bash
python train_align.py \
    --model_dir path/to/qwen \
    --adapter_path path/to/adapter \
    --version_name qwen2_5vl_reproduce_align \
    --num_epochs 6 \
    --lr 5e-5 \
    --accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --sample_num 100 \
    --forward_nums 4 \
    --weight_decay 1.0 \
    --top_k 1 \
    --coco_path path/to/coco \
    --sharegpt4v_caption_path path/to/sharegpt4vcaption \
    --videor1_cot_dataset path/to/pics \
    --videor1_cot_165k path/to/cot file
```

Finally evaluate the draft model on MMMU:

```bash
python mmmu-eval.py \
    --model_path path/to/qwen   \
    --dataset_path path/wo/mmmu  \
    --adapter_path path/to/adapter    \
    --draft_length 5    \
    --draft_k 8 \
    --draft_total_token 63 \
    --sample_num 10 
```