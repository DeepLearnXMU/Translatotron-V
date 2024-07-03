## Stage 1
First you will need to train your image tokenizer

```
python -m torch.distributed.launch --nproc_per_node=8 train_mgpu.py \
    --output_dir $save_name \
    --vq_codebook_size $vq_codebook_size \
    --vq_codebook_dim $vq_codebook_dim \
    --data_dir $data_dir \
    --patch_size 16 \
    --dim $dim \
    --num_layers $num_layers \
    --batch_size $batch_size \
    --grad_accum_every $grad_accum_every
```

## Stage 2

Then train the IIMT model

```
python -m torch.distributed.launch --nproc_per_node=8 run_translatotron_v.py \
    --train_lmdb_path $train_dir \
    --valid_lmdb_path $valid_dir \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 2 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type polynomial\
    --weight_decay 0.0001 \
    --seed 42 \
    --src_lang de \
    --tgt_lang en \
    --output_dir $save_name \
    --src_tokenizer_path config/char_de.tokenizer \
    --tgt_tokenizer_path config/char_en.tokenizer \
    --vae_config_path config/vit_vqgan_8192cb.json \
    --iit_config_path config/iit_transformer_512dim.json \
    --teacher_model_weight $teacher_path \
    --teacher_config_path config/t2i_transformer_distill.json \
    --temperature 1.0 \
    --vae_weight $vae_path \
    --use_amp true \
    --num_workers 32 \
    --checkpointing_steps epoch
```

## Inference

```
python -m torch.distributed.launch --nproc_per_node=8 run_test.py \
    --test_lmdb_path $test_dir \
    --per_device_test_batch_size 8 \
    --src_lang de \
    --tgt_lang en \
    --output_dir $output_dir \
    --src_tokenizer_path config/char_de.tokenizer \
    --tgt_tokenizer_path config/char_en.tokenizer \
    --vae_config_path config/vit_vqgan_8192cb.json \
    --iit_config_path config/iit_transformer_512dim.json \
    --model_weights_path $model_path \
    --vae_weight $vae_path \
    --use_amp true \
    --num_workers 16
```