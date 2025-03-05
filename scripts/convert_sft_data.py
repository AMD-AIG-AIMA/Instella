import os
import logging
from argparse import ArgumentParser

import datasets as ds

from instella.util import prepare_cli_environment
from tqdm import tqdm

import gzip
import json
from tqdm import trange

log = logging.getLogger(__name__)


def convert_code_feedback(dataset, mix=False):
    log.info("Converting code_feedback ...")
    y_all = []
    for i, sample in enumerate(dataset):
        y = {
            "dataset": "code_feedback",
            "id": "code_feedback_{}".format(i),
            "messages": sample['messages']
        }
        y_all.append(y)

    log.info(f"In total {len(y_all)} samples")
    return y_all


def convert_OpenHermes(dataset, mix=False):
    log.info("Converting OpenHermes ...")
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    y_all = []
    for i, sample in enumerate(dataset):
        y = {
            "dataset": "OpenHermes",
            "id": "OpenHermes_{}".format(i),
            "messages": [{"role": role_map[mssg["from"]], "content": mssg["value"]} for mssg in sample['conversations']]
        }
        y_all.append(y)
    
    log.info(f"In total {len(y_all)} samples")
    return y_all


def convert_WebInstructSub(dataset, mix=False):
    log.info("Converting WebInstructSub ...")
    y_all = []
    for i, sample in tqdm(enumerate(dataset)):
        y = {
            "dataset": "WebInstructSub",
            "id": "WebInstructSub_{}".format(i),
            "messages": [{"role": "user", "content": sample["question"]}, {"role": "assistant", "content": sample["answer"]}]
        }
        y_all.append(y)
    
    log.info(f"In total {len(y_all)} samples")
    return y_all
    
    
def main(opts) -> None:
    if opts.dataset == "tulu-3":
        dataset = ds.load_dataset("allenai/tulu-3-sft-mixture", split="train")
    elif opts.dataset == "ultrachat-200k":
        dataset = ds.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    elif opts.dataset == "code-feedback":
        dataset = ds.load_dataset("m-a-p/Code-Feedback", split="train")
        dataset = convert_code_feedback(dataset, mix=True)
    elif opts.dataset == "OpenHermes":
        dataset = ds.load_dataset("teknium/OpenHermes-2.5", split="train")
        dataset = convert_OpenHermes(dataset, mix=True)
    elif opts.dataset == "WebInstructSub":
        dataset = ds.load_dataset("TIGER-Lab/WebInstructSub", split="train")
        dataset = convert_WebInstructSub(dataset, mix=True)
    elif opts.dataset == "instella-gsm8k-synthetic":
        dataset = ds.load_dataset("amd/Instella-GSM8K-synthetic", split="train_119K")

    os.makedirs(opts.output_dir, exist_ok=True)
    file_path = f'{opts.output_dir}/data.jsonl.gz'
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        for i in trange(len(dataset)):
            example = dataset[i]
            content = preprocess(example, opts.dataset)
            # Convert each dictionary to a JSON string and write it to the file
            record = {
                "id": f"{opts.dataset}-{i}",
                "text": content,
                "source": opts.dataset,
            }
            json.dump(record, f)
            f.write('\n')  # Ensure each JSON object is on a new line

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(example, dataset):
    if dataset == "ultrachat":
        content = "\n".join(example["data"])
    else:   
        content = ""
        for msg in example["messages"]:
            message = msg["content"].strip() + "\n"
            content += message
    return content


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Convert SFT dataset for pretraining")
    parser.add_argument("--output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument("-ds", "--dataset", type=str, help="""Dataset that we are processing""", default="tulu-3")
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)