import sys
import nltk
import evaluate
import sacrebleu
import argparse

def read_file(path):
    i = 0
    toks = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            toks.append(line)
            i += 1
    return toks, i

def read_ref_file(path):
    i = 0
    toks = []
    with open(path) as f:
        for line in f.readlines():
            if "ref_tgt:" in line:
                line = line.strip()
                toks.append(line[9:])
                i += 1
    return toks, i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Where to store the final model.")
    args = parser.parse_args()
    sys_toks, i1 = read_file(f"{args.input_dir}/generate_img/generate.txt")
    ref_toks, i2 = read_ref_file(f"{args.input_dir}/test.txt")

    assert i1 == i2, "error"

    sys_translations, refs = [], []
    cor_num, err_num = 0 , 0
    for k in range(i1):
        sys_translations.append(sys_toks[k])
        refs.append(ref_toks[k])


    result = sacrebleu.corpus_bleu(sys_translations,[refs])

    print("ref sacrebleu: {}".format(result.score))

