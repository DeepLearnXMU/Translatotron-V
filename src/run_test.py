# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any Transformers model for image translation leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from accelerate import DistributedDataParallelKwargs
from numpy import mean
from tqdm.auto import tqdm
from einops import rearrange
from tokenizers import Tokenizer

from transformers import PreTrainedTokenizerFast
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import easyocr

from parti_pytorch import TITImageDataset, TranslatotronV, TITImageTextDataset, TITImageTextLmdbDataset, BeamSearchScorer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0")

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TIT model on an image dataset")
    parser.add_argument("--test_lmdb_path", type=str, default=None, help="A folder containing the image test data.")
    parser.add_argument("--image_size", type=int, default=256, help="The size of the input images.")
    parser.add_argument("--src_lang", type=str, default=None, help="The source language.")
    parser.add_argument("--tgt_lang", type=str, default=None, help="The target language.")
    parser.add_argument("--max_src_length", type=int, default=1024, help="The max length of input text")
    parser.add_argument("--max_tgt_length", type=int, default=1024, help="The max length of output text")
    parser.add_argument("--src_tokenizer_path", type=str, default=None, help="The path of source tokenizer.")
    parser.add_argument("--tgt_tokenizer_path", type=str, default=None, help="The path of target tokenizer.")
    parser.add_argument("--vae_config_path", type=str, default=None, help="The path of vae config file.")
    parser.add_argument("--iit_config_path", type=str, default=None, help="The path of iit config file.")
    parser.add_argument("--vae_weight", type=str, default=None, help="The path of vit_vqgan weight.")
    parser.add_argument("--num_workers", type=int, default=1, help="num of workers for dataloader.")
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )

    parser.add_argument(
        "--use_amp", type=bool, default=False, help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default=None,
        help="The path of the model to load from local file system or remote (e.g. HF hub) repo.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="For distributed training: local_rank",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.test_lmdb_path is None:
        raise ValueError("Need a dataset name or a test folder.")

    if args.output_dir is None:
        raise ValueError(
            "Need an `output_dir` to create a repo."
        )

    if args.src_lang is None or args.tgt_lang is None:
        raise ValueError("Need to specify both source and target languages.")
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(mixed_precision="fp16" if args.use_amp else None, 
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], **accelerator_log_kwargs)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load config from json file
    with open(args.vae_config_path, "r") as f:
        vae_config = json.load(f)
        
    with open(args.iit_config_path, "r") as f:
        iit_config = json.load(f)
        
    image_size = vae_config["image_size"] if "image_size" in vae_config else args.image_size
    
    if args.src_tokenizer_path is not None:
        src_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.src_tokenizer_path)
    else:
        raise ValueError("Need to specify a source tokenizer.")

    if args.tgt_tokenizer_path is not None:
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tgt_tokenizer_path)
    else:
        raise ValueError("Need to specify a target tokenizer.")
    
    model = TranslatotronV(**iit_config, patch_size = vae_config['patch_size'], img_size = image_size, vae_config = vae_config, 
                           vae_weight = args.vae_weight, src_text_tokenizer = src_tokenizer, tgt_text_tokenizer = tgt_tokenizer)
    model.load_state_dict(torch.load(args.model_weights_path, map_location = "cpu"))
    # Define torchvision transforms to be applied to each image.
    test_transforms = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    
    test_src_transforms = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(224),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
    
    # Get the datasets
    test_dataset = TITImageTextLmdbDataset(args.src_lang, args.tgt_lang, args.test_lmdb_path, image_size, test_transforms)

    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))

    
    def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        return shifted_input_ids
    
    def collate_fn(batch):
        src_img, tgt_img, src_text_label, tgt_text_label = [], [], [], []
        for i in range(len(batch)):
            src_img.append(batch[i][0])
            tgt_img.append(batch[i][1])
            # src_text.append(src_tokenizer.bos_token + batch[i][2])
            src_text_label.append(batch[i][2] + src_tokenizer.eos_token)
            # tgt_text.append(src_tokenizer.bos_token + batch[i][3])
            tgt_text_label.append(batch[i][3] + tgt_tokenizer.eos_token)
            
        # coverting to tensors
        src_img = torch.stack(src_img)
        tgt_img = torch.stack(tgt_img)
        src_text_label = src_tokenizer.batch_encode_plus(src_text_label, truncation=True, padding=True, max_length=args.max_src_length, return_tensors="pt")
        src_text_label['labels'] = src_text_label['input_ids']
        src_text_label['input_ids'] = shift_tokens_right(src_text_label['input_ids'], src_tokenizer.bos_token_id)
        src_text_label['attention_mask'] = src_text_label['attention_mask'].bool()
        
        
        tgt_text_label = tgt_tokenizer.batch_encode_plus(tgt_text_label, truncation=True, padding=True, max_length=args.max_tgt_length, return_tensors="pt")
        tgt_text_label['labels'] = tgt_text_label['input_ids']
        tgt_text_label['input_ids'] = shift_tokens_right(tgt_text_label['input_ids'], tgt_tokenizer.bos_token_id)
        tgt_text_label['attention_mask'] = tgt_text_label['attention_mask'].bool()
        return src_img, tgt_img, src_text_label, tgt_text_label
    
    # train_dataset.map(collate_fn, batched=True, num_proc=args.num_workers)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)


    # Scheduler and math around the number of training steps.

    # Prepare everything with our `accelerator`.
    model, test_dataloader= accelerator.prepare(model, test_dataloader)

    # Get the metric function
    # metric = evaluate.load("accuracy")

    # Test!
    total_batch_size = args.per_device_test_batch_size * accelerator.num_processes
    generate_img_dir = args.output_dir + "/generate_img/"
    ref_img_dir = args.output_dir + "/ref_img/"
    if accelerator.is_local_main_process:
        if not os.path.exists(generate_img_dir):
            os.makedirs(generate_img_dir)
        if not os.path.exists(ref_img_dir):
            os.makedirs(ref_img_dir)
            
            
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_test_batch_size}")
    logger.info(f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    

    model.eval()
    all_accuracy = []
    count = 0
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            images, image_tokens, tgt_texts  = accelerator.unwrap_model(model).generate(batch[0])
            _, ref_image_tokens, _ = accelerator.unwrap_model(model).vae.encode(batch[1], return_indices_and_loss = True)
            src_texts = accelerator.unwrap_model(model).generate_text(batch[0])
            ref_src_texts = batch[2]['labels']
            ref_tgt_texts = batch[3]['labels']

        images, sources, references, image_tokens, ref_image_tokens = accelerator.gather_for_metrics((images, batch[0], batch[1], 
                                                                                        image_tokens, ref_image_tokens
                                                                                        ))
        src_texts = accelerator.gather_for_metrics((src_texts))
        ref_src_texts = accelerator.pad_across_processes(ref_src_texts, dim=1, pad_index=src_tokenizer.pad_token_id)
        ref_tgt_texts = accelerator.pad_across_processes(ref_tgt_texts, dim=1, pad_index=tgt_tokenizer.pad_token_id)
        tgt_texts = accelerator.pad_across_processes(tgt_texts, dim=1, pad_index=tgt_tokenizer.pad_token_id)
        ref_src_texts, ref_tgt_texts, tgt_texts = accelerator.gather_for_metrics((ref_src_texts, ref_tgt_texts, tgt_texts))
        
        accuracy = ((image_tokens==ref_image_tokens).sum()/image_tokens.numel()).cpu()
        
        all_accuracy.append(accuracy)
        # save the images
        if accelerator.is_local_main_process:
            # save pil images
            for i in range(len(images)):
                save_image(images[i], generate_img_dir + f'{count}.jpg', normalize = True, value_range = (0, 1))
                save_image(references[i], ref_img_dir + f'{count}.jpg', normalize = True, value_range = (0, 1))
                count += 1

        accelerator.print("accuracy:{}".format(accuracy))
        src_texts = src_tokenizer.batch_decode(src_texts, skip_special_tokens=True)
        tgt_texts = tgt_tokenizer.batch_decode(tgt_texts, skip_special_tokens=True)
        ref_src_texts = src_tokenizer.batch_decode(ref_src_texts, skip_special_tokens=True)
        ref_tgt_texts = tgt_tokenizer.batch_decode(ref_tgt_texts, skip_special_tokens=True)
        
        # save the text into file
        if accelerator.is_local_main_process:
            with open(args.output_dir + "/" + f'test.txt', 'a') as f:
                for i in range(len(src_texts)):
                    f.write("src: " + src_texts[i] + "\n")
                    f.write("tgt: " + tgt_texts[i] + "\n")
                    f.write("ref_src: " + ref_src_texts[i] + "\n")
                    f.write("ref_tgt: " + ref_tgt_texts[i] + "\n")
                    f.write("\n")
        

        accelerator.print("all accuracy:{}".format(mean(all_accuracy)))




if __name__ == "__main__":
    main()