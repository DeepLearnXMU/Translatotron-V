
export prefix="your scr code directory"
export save_name="model save name"

/opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=8 --master_port=27699 $prefix/src/run_t2i_with_layout.py \
    --train_lmdb_path $prefix/data-build/iwslt14.de-en-lmdb/train_ \
    --valid_lmdb_path $prefix/data-build/iwslt14.de-en-lmdb/valid_ \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type polynomial\
    --weight_decay 0.0001 \
    --seed 42 \
    --src_lang de \
    --tgt_lang en \
    --output_dir $prefix/result_new/$save_name \
    --tgt_tokenizer_path $prefix/src/config/char_en.tokenizer \
    --vae_config_path $prefix/src/config/vit_vqgan_8192cb.json \
    --t2i_config_path $prefix/src/config/t2i_transformer_distill.json \
    --vae_weight $prefix/image-tokenizer/en/vae.pt \
    --resume_from_checkpoint $prefix/result_new/$save_name/epoch_6 \
    --use_amp true \
    --num_workers 16 \
    --checkpointing_steps epoch \
    >>$prefix/log_latest/$save_name.log 2>&1