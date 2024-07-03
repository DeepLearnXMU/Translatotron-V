import lmdb
import pickle
from pathlib import Path
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import os

def process_data(args):
    path, text, idx = args
    with open(path, 'rb') as f:
        image_data = f.read()
    return idx, pickle.dumps((str(text), image_data))


def create_lmdb_dataset(paths, texts, write_frequency=5000):
    env = lmdb.open(lmdb_path, map_size=1099511627776 * 2)  # 设置足够大的map_size以存储所有数据

    with Pool(processes=args.num_workers) as pool:
        txn = env.begin(write=True)
        for i in tqdm(range(0, len(paths), write_frequency)):
            batch = list(zip(paths[i:i+write_frequency], texts[i:i+write_frequency], range(i, min(i+write_frequency, len(paths)))))
            results = pool.map(process_data, batch)

            for idx, data in results:
                txn.put(f"{idx}".encode(), data)
            txn.commit()
            txn = env.begin(write=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--image_data_dir", type=str, default='data-build/wmt14-images', help="Where to load data.")
    parser.add_argument("--text_data_dir", type=str, default='data-build/wmt14', help="Where to load data.")
    parser.add_argument("--src_lang", type=str, default='de', help="Source language.")
    parser.add_argument("--tgt_lang", type=str, default='en', help="Target language.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers.")

    args = parser.parse_args()

    for split in ['train','valid','test']:
        for lang in [args.src_lang, args.tgt_lang]:
            folder = f'{args.image_data_dir}/{split}_{lang}'
            text_file = f'{args.text_data_dir}/{split}.{lang}'
            lmdb_path = f'{args.output_dir}/{split}_{lang}'
            if os.path.exists(lmdb_path) is False:
                os.makedirs(lmdb_path)
            exts = ['jpg', 'jpeg', 'png']
            paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
            texts = []
            # read text file
            with open(text_file, 'r') as f:
                temp_texts = f.readlines()
                for path in paths:
                    texts.append(temp_texts[int(path.name[:path.name.find(".")])].strip().lower())
            create_lmdb_dataset(paths, texts)