# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import yaml
import torch
import joblib

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from litgpt.utils import CLI

import models.vqvae as vqvae
import options.option_vq as option_vq



def load_data(file_path, checkpoints_path, dataset_dir):
    with open(file_path, 'rb') as f:
        ori_data = joblib.load(f)
    args = option_vq.get_args_parser()
    args.dataname = 'humanml'
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        512,
                        args.code_dim,
                        args.output_emb_width,
                        2,
                        args.stride_t,
                        args.width,
                        3,
                        3,
                        'relu',
                        args.vq_norm)
    state_dict = torch.load(checkpoints_path)
    net.load_state_dict(state_dict['net'])
    net.cuda()
    mean = np.load(os.path.join(dataset_dir, 'Mean.npy'))
    std = np.load(os.path.join(dataset_dir, 'Std.npy'))

    nums = len(ori_data['input_motion'])
    data = []
    for i in tqdm(range(nums)):
        input_motion = ori_data['input_motion'][i][0][:ori_data['lengths'][i]].cpu().numpy()
        output_motion = ori_data['motion'][i][0][:ori_data['lengths'][i]]
        input_motion = (input_motion - mean) / std
        output_motion = (output_motion - mean) / std

        input_token = net.encode(torch.tensor(input_motion).unsqueeze(0).float().cuda()).reshape(-1).int().detach().cpu().numpy().tolist()
        output_token = net.encode(torch.tensor(output_motion).unsqueeze(0).float().cuda()).reshape(-1).int().detach().cpu().numpy().tolist()

        caption = ori_data['text'][i]
        assert 'Upper body: A person' in caption or 'Lower body: A person' in caption
        if 'Upper body: A person' in caption:
            caption = caption[:-1] + ' with the upper body.'
        else:
            caption = caption[:-1] + ' with the lower body.'

        caption = caption.replace('Upper body: A person', 'The person').replace('Lower body: A person', 'The person')

        data.append({
                'caption': caption,
                'input_token': input_token,
                'output_token': output_token,
                'idx': i,
                'input_motion': input_motion,
                'output_motion': output_motion,
                'length': ori_data['lengths'][i],
        })
    return data



def prepare(
    destination_path: Path,
    encoder_checkpoints_path: str,
    edit_results_path: str,
    dataset_dir: str,
    mask_inputs: bool = False,  # as in alpaca-lora
    ignore_index: int = -100,
    max_seq_length: int = 1024,
) -> None:
    """Prepare the Alpaca dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading data file...")

    dataset = load_data(edit_results_path, encoder_checkpoints_path, dataset_dir)

    print(f"Dataset has {len(dataset):,} samples")

    print("Processing dataset ...")
    dataset = [
        prepare_sample(
            example=sample,
            tokenizer=None,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(dataset)
    ]
    joblib.dump(dataset, destination_path / "alpaca.pkl")



def prepare_sample(example: dict) -> dict:
    return(generate_prompt(example))


def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    input_person = '['
    for token in example['input_token']:
        input_person += str(token) + ' '
    input_person = input_person[:-1] + ']'
    output_person = '['
    for token in example['output_token']:
        output_person += str(token) + ' '
    output_person = output_person[:-1] + ']'
    return ({
        "instruction": f"I will give you two motion sequences, representing sequences of the same character doing different actions. You are asked to compare two sequences and output what modifications the person make to transfer from the first action to the second action.",
        "input": f"Action 1:\n{input_person}\n\n### Action 2:\n{output_person}",
        "output": example["caption"],
        "idx": example["idx"],
        'input_motion': example['input_motion'],
        'output_motion': example['output_motion'],
        'length': example['length'],

    })


if __name__ == "__main__":
    CLI(prepare)
