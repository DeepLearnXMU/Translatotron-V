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

import easyocr
import sacrebleu
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

from parti_pytorch import TITImageDataset, TITImageTextLmdbDataset,T2IBertLayoutTransformer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0")

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TIT model on an image dataset")
    parser.add_argument("--train_lmdb_path", type=str, default=None, help="A folder containing the image training data.")
    parser.add_argument("--valid_lmdb_path", type=str, default=None, help="A folder containing the image validation data.")
    parser.add_argument("--image_size", type=int, default=256, help="The size of the input images.")
    parser.add_argument("--src_lang", type=str, default=None, help="The source language.")
    parser.add_argument("--tgt_lang", type=str, default=None, help="The target language.")
    parser.add_argument("--max_src_length", type=int, default=1024, help="The max length of input text")
    parser.add_argument("--max_tgt_length", type=int, default=1024, help="The max length of output text")
    parser.add_argument("--tgt_tokenizer_path", type=str, default=None, help="The path of target tokenizer.")
    parser.add_argument("--vae_config_path", type=str, default=None, help="The path of vae config file.")
    parser.add_argument("--t2i_config_path", type=str, default=None, help="The path of iit config file.")
    parser.add_argument("--vae_weight", type=str, default=None, help="The path of vit_vqgan weight.")
    parser.add_argument("--num_workers", type=int, default=1, help="A folder containing the validation data.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
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
    if args.train_lmdb_path is None and args.valid_lmdb_path is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

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

    accelerator = Accelerator(mixed_precision="fp16" if args.use_amp else None, gradient_accumulation_steps=args.gradient_accumulation_steps, 
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
        
    with open(args.t2i_config_path, "r") as f:
        t2i_config = json.load(f)
        
    image_size = vae_config["image_size"] if "image_size" in vae_config else args.image_size
    

    if args.tgt_tokenizer_path is not None:
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tgt_tokenizer_path)
    else:
        raise ValueError("Need to specify a target tokenizer.")
    
    model = T2IBertLayoutTransformer(**t2i_config, patch_size = vae_config['patch_size'], img_size = image_size, vae_config = vae_config, 
                           vae_weight = args.vae_weight, tgt_text_tokenizer = tgt_tokenizer)
    # Define torchvision transforms to be applied to each image.
    train_transforms = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            # T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    train_src_transforms = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(224),
        # T.RandomHorizontalFlip(),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
    val_transforms = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
    
    val_src_transforms = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(224),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
    
    # Get the datasets
    train_dataset = TITImageTextLmdbDataset(args.src_lang, args.tgt_lang, args.train_lmdb_path, image_size, train_transforms)
    eval_dataset = TITImageTextLmdbDataset(args.src_lang, args.tgt_lang, args.valid_lmdb_path, image_size, val_transforms)
    
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        return shifted_input_ids
    
    def collate_fn(batch):
        src_img, tgt_text_label, tgt_img = [], [], []
        for i in range(len(batch)):
            src_img.append(batch[i][0])
            tgt_img.append(batch[i][1])
            tgt_text_label.append(batch[i][3] + tgt_tokenizer.eos_token)
            
        # coverting to tensors
        src_img = torch.stack(src_img)
        tgt_img = torch.stack(tgt_img)
        
        tgt_text_label = tgt_tokenizer.batch_encode_plus(tgt_text_label, truncation=True, padding=True, max_length=args.max_tgt_length, return_tensors="pt")
        tgt_text_label['labels'] = tgt_text_label['input_ids']
        # tgt_text_label['input_ids'] = shift_tokens_right(tgt_text_label['input_ids'], tgt_tokenizer.bos_token_id)
        tgt_text_label['attention_mask'] = tgt_text_label['attention_mask'].bool()
        return src_img, tgt_text_label, tgt_img
    
    # train_dataset.map(collate_fn, batched=True, num_proc=args.num_workers)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("run_tit", experiment_config)

    # Get the metric function
    # metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint,strict=False)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            progress_bar.update((starting_epoch - 1) * len(train_dataloader))
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    
    # reader = easyocr.Reader([args.tgt_lang], gpu=accelerator.device)
    # best_bleu = 0
    best_accuracy = 0
    patience_count = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                loss = model(src_images=batch[0], tgt_text_input=batch[1], tgt_images=batch[2], return_loss=True)
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch: {epoch} Loss: {loss.detach().float()}")
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        all_accuracy = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                images, image_tokens  = accelerator.unwrap_model(model).generate(batch[0],batch[1])
                _, ref_image_tokens, _ = accelerator.unwrap_model(model).vae.encode(batch[2], return_indices_and_loss = True)

            images, references, image_tokens, ref_image_tokens = accelerator.gather_for_metrics((images, batch[2], 
                                                                                          image_tokens, ref_image_tokens
                                                                                          ))

            accuracy = ((image_tokens==ref_image_tokens).sum()/image_tokens.numel()).cpu()
            all_accuracy.append(accuracy)
            imgs_and_recons = torch.stack((references, images), dim = 0)
            imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
            imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
            grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))
            
            if accelerator.is_local_main_process:
                save_image(grid, (args.output_dir + "/" + f'epoch_{str(epoch)}.png'))                
                

        accelerator.print("epoch {}: all accuracy: {}, best accuracy: {}".format(epoch, mean(all_accuracy), best_accuracy))
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": mean(all_accuracy),
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            if epoch < args.num_train_epochs / 2:
                continue
            if mean(all_accuracy) > best_accuracy:
                best_accuracy = mean(all_accuracy)
                patience_count = 0
            else:
                if best_accuracy != 0:
                    patience_count += 1
                # early stop
                if patience_count > args.patience:
                    break

    if args.with_tracking:
        accelerator.end_training()

    # average checkpoins
    if accelerator.is_local_main_process:
        all_folders = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]
        # sort folders by epoch
        all_folders = sorted(all_folders, key=lambda x: int(x.split("_")[1]))

        # get the last folders
        num_epochs = args.patience

        last_10_folders = all_folders[-num_epochs:]

        # load the state_dict from the first folder
        first_state_dict = torch.load(os.path.join(args.output_dir, last_10_folders[0], 'pytorch_model.bin'))
        for key in first_state_dict.keys():
            if first_state_dict[key].dtype is not torch.int64 and first_state_dict[key].dtype is not torch.int32 :
                first_state_dict[key] *= 1.0 / num_epochs

        # sum the state_dicts from all other folders
        for folder in last_10_folders[1:]:
            current_state_dict = torch.load(os.path.join(args.output_dir, folder, 'pytorch_model.bin'))
            for key in first_state_dict.keys():
                if first_state_dict[key].dtype is not torch.int64 and first_state_dict[key].dtype is not torch.int32:
                    first_state_dict[key] += current_state_dict[key] / num_epochs

        # save the averaged state_dict to disk
        accelerator.unwrap_model(model).load_state_dict(first_state_dict)
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, 'average_pytorch_model.bin'))


if __name__ == "__main__":
    main()