
export prefix="your scr code directory"
export save_name="model save name"


python -m torch.distributed.launch --nproc_per_node=8 $prefix/src/run_translatotron_v.py \
    --train_lmdb_path $prefix/data-build/iwslt14.de-en-lmdb/train_ \
    --valid_lmdb_path $prefix/data-build/iwslt14.de-en-lmdb/valid_ \
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
    --output_dir $prefix/rseult/$save_name \
    --src_tokenizer_path $prefix/src/config/char_de.tokenizer \
    --tgt_tokenizer_path $prefix/src/config/char_en.tokenizer \
    --vae_config_path $prefix/src/config/vit_vqgan_8192cb.json \
    --iit_config_path $prefix/src/config/iit_transformer_512dim.json \
    --teacher_model_weight $prefix/result_new/t2i_layout_avg/average_pytorch_model.bin \
    --teacher_config_path $prefix/src/config/t2i_transformer_distill.json \
    --temperature 1.0 \
    --vae_weight $prefix/image-tokenizer/en/vae.pt \
    --use_amp true \
    --num_workers 32 \
    --max_eval_samples 500 \
    --checkpointing_steps epoch
