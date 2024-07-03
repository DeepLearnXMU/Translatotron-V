import os
from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import multiprocessing as mp
import argparse

def split_image_into_patches(img, patch_size):
    patches = []
    height, width, channels = img.shape
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def calculate_ssim_for_files(g_file, r_file, patch_size):
    g_img = cv2.imread(g_file)
    r_img = cv2.imread(r_file)

    g_patches = split_image_into_patches(g_img, patch_size)
    r_patches = split_image_into_patches(r_img, patch_size)

    patch_ssims = []
    for g_patch, r_patch in zip(g_patches, r_patches):
        patch_ssim = compare_ssim(g_patch, r_patch, channel_axis=-1, multichannel=True)
        patch_ssims.append(patch_ssim)

    return np.mean(patch_ssims)

def process_files(file_pair):
    g_file, r_file = file_pair
    ssims = {}
    for patch_size in patch_sizes:
        ssims[patch_size] = calculate_ssim_for_files(g_file, r_file, patch_size)
    return ssims

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, help="Where to store the final model.")
    args = parser.parse_args()

    test_result_folder = "{}/generate_img".format(args.input_dir)

    test_result_files = os.listdir(test_result_folder)
    test_result_files = [file for file in test_result_files if 'jpg' in file]

    test_result_files = sorted(test_result_files, key=lambda x: int(x.split('.')[0]))
    test_result_files = [os.path.join(test_result_folder, file) for file in test_result_files]
    
    ref_result_folder = "{}/ref_img".format(args.input_dir)

    ref_result_files = os.listdir(ref_result_folder)
    ref_result_files = [file for file in ref_result_files if 'jpg' in file]

    ref_result_files = sorted(ref_result_files, key=lambda x: int(x.split('.')[0]))
    ref_result_files = [os.path.join(ref_result_folder, file) for file in ref_result_files]
    
    psnrs = []
    ssims = []
    mses = []
    patch_sizes = [512]
    ssims = {patch_size: [] for patch_size in patch_sizes}
    print(mp.cpu_count())
    with mp.Pool(64) as pool:
        for file_ssims in tqdm(pool.imap(process_files, zip(test_result_files, ref_result_files)), total=len(test_result_files)):
            for patch_size in patch_sizes:
                ssims[patch_size].append(file_ssims[patch_size])
    average_ssims = {patch_size: np.mean(ssims[patch_size]) for patch_size in patch_sizes}

    for patch_size in patch_sizes:
        print(f"Average SSIM for patch {patch_size}: {average_ssims[patch_size]}")
    