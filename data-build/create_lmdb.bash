
export prefix="your scr code directory"


export save_name=data-build/iwslt17.ro-en-lmdb

/opt/conda/bin/python $prefix/data-build/create_lmdb_mulproc.py \
    --output_dir $prefix/$save_name \
    --text_data_dir $prefix/data-build/iwslt17.ro-en \
    --image_data_dir $prefix/data-build/iwslt17.ro-en-images \
    --src_lang ro \
    --tgt_lang en \
    --num_workers 64