import easyocr
import os
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--lang", type=str, default='en', help="Source language.")

    args = parser.parse_args()
    reader = easyocr.Reader([args.lang]) # this needs to run only once to load the model into memory

    test_result_folder = args.input_dir
    
    test_result_files = os.listdir(test_result_folder)
    test_result_files = [file for file in test_result_files if 'jpg' in file]

    test_result_files = sorted(test_result_files, key=lambda x: int(x.split('.')[0]))
    test_result_files = [os.path.join(test_result_folder, file) for file in test_result_files]

    # write result into file
    count = 0
    with open(os.path.join(test_result_folder, 'generate.txt'), 'w') as f:
        for file in tqdm(test_result_files):
            ocr_result = reader.readtext(file, paragraph=True)
            ocr_result = [item[1] for item in ocr_result]
            ocr_result = ' '.join(ocr_result)
            f.write(ocr_result + '\n')
            print(count)
            count += 1    
