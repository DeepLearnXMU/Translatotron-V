
export prefix="your scr code directory"
export save_name="model save name"

python -m torch.distributed.launch --nproc_per_node=8 $prefix/src/run_test.py \
    --test_lmdb_path $prefix/data-build/iwslt14.de-en-lmdb/test_ \
    --per_device_test_batch_size 16 \
    --src_lang de \
    --tgt_lang en \
    --output_dir $prefix/result/${save_name}_test \
    --src_tokenizer_path $prefix/src/config/char_de.tokenizer \
    --tgt_tokenizer_path $prefix/src/config/char_en.tokenizer \
    --vae_config_path $prefix/src/config/vit_vqgan_8192cb.json \
    --iit_config_path $prefix/src/config/iit_transformer_512dim.json \
    --model_weights_path $prefix/result/$save_name/average_pytorch_model.bin \
    --vae_weight $prefix/image-tokenizer/en/vae.pt \
    --use_amp true \
    --num_workers 16

python $prefix/eval/ocr.py \
    --lang en \
    --input_dir $prefix/result/${save_name}_test/generate_img

python $prefix/eval/ocr.py \
    --lang en \
    --input_dir $prefix/result/${save_name}_test/ref_img

python $prefix/eval/cal_ocr_ref_bleu.py \
    --input_dir $prefix/result/${save_name}_test \
    >>$prefix/log/${save_name}_test.log 2>&1

python $prefix/eval/structure_bleu.py \
    --generate_dir $prefix/result/${save_name}_test/generate_img \
    --ref_dir $prefix/result/${save_name}_test/ref_img \
    --lang en \
    >>$prefix/log/${save_name}_test.log 2>&1

python $prefix/eval/img_eval.py \
    --input_dir $prefix/result/${save_name}_test \
    >> $prefix/log/${save_name}_test.log 2>&1