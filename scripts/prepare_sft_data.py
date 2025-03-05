"""
Script for preparing the SFT data for fine-tuning Instella model.
Modifed from https://github.com/allenai/OLMo/blob/main/scripts/prepare_tulu_data.py
"""

import os

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from instella.tokenizer import Tokenizer
from instella.util import prepare_cli_environment
import random
from tqdm import tqdm

log = logging.getLogger(__name__)


def convert_mmlu(dataset):
    # Define question templates and their matching answer templates
    qa_templates = [
        {
            "question": "Given the following question and four candidate answers (A, B, C, and D), select the most appropriate answer.\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The most appropriate answer is {answer}."
        },
        {
            "question": "Choose the correct answer from the options below (A, B, C, or D):\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The correct answer is {answer}."
        },
        {
            "question": "Based on the following question and options, identify the best answer (A, B, C, or D).\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The best answer is {answer}."
        },
        {
            "question": "Analyze the question and the four answer choices (A, B, C, and D). Then select the correct answer.\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "After analysis, the correct answer is {answer}."
        },
        {
            "question": "Consider the question below and choose the best option from A, B, C, or D.\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The best option is {answer}."
        },
        {
            "question": "Select the correct response (A, B, C, or D) to the question provided below.\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The correct response is {answer}."
        },
        {
            "question": "Determine the correct choice (A, B, C, or D) based on the question and information provided.\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The determined correct choice is {answer}."
        },
        {
            "question": "From the following question and answer choices, identify the correct option (A, B, C, or D).\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The identified correct option is {answer}."
        },
        {
            "question": "Using the given question and four possible answers, select the best option (A, B, C, or D).\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The selected best option is {answer}."
        },
        {
            "question": "Read the question and options below, then determine the correct answer (A, B, C, or D).\nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "The determined correct answer is {answer}."
        },
        {
            "question": "Question: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}",
            "answer": "{answer}."
        },
    ]

    # Function to generate a question and answer pair
    def generate_question_answer(question, options, correct_answer, shuffle=True):
        # Unpack and shuffle options with their corresponding labels
        labels = ['A', 'B', 'C', 'D']
        correct_option = options[correct_answer]
        if shuffle:
            random.shuffle(options)

        # Extract the shuffled options and update the correct answer's label
        shuffled_options = {label: option for label, option in zip(labels, options)}
        new_answer_label = next(
            label for label, option in shuffled_options.items() if option == correct_option
        )

        # Randomly select a QA template
        qa_template = random.choice(qa_templates)

        # Format the question and answer
        formatted_question = qa_template["question"].format(
            question=question,
            option_a=shuffled_options["A"],
            option_b=shuffled_options["B"],
            option_c=shuffled_options["C"],
            option_d=shuffled_options["D"],
        )
        formatted_answer = qa_template["answer"].format(answer=new_answer_label)

        # Combine question and answer
        return {"question": formatted_question, "answer": formatted_answer}
    
    log.info("Converting mmlu ...")
    y_all = []
    for i, sample in enumerate(dataset):
        qa_pair = generate_question_answer(sample['train']['question'],  sample['train']['choices'],  sample['train']['answer'])
        y = {
            "dataset": "mmlu",
            "id": "mmlu_{}".format(i),
            "messages": [{"role": "user", "content": qa_pair['question']}, {"role": "assistant", "content": qa_pair['answer']}]
        }
        y_all.append(y)

    log.info(f"In total {len(y_all)} samples")

    return ds.Dataset.from_list(y_all)

def convert_o1journey(dataset):
    log.info("Converting o1 journey ...")
    y_all = []
    for i, sample in enumerate(dataset):
        y = {
            "dataset": "o1_journey",
            "id": "o1_journey_{}".format(i),
            "messages": [{"role": "user", "content": sample["question"]}, {"role": "assistant", "content": sample["longCOT"].split("\n####")[0]}]
        }
        y_all.append(y)
    
    # upsample 10x
    y_all = y_all * 10 
    random.shuffle(y_all)
    log.info(f"In total {len(y_all)} samples")
    return ds.Dataset.from_list(y_all)

def convert_openmathinstruct(dataset):
    log.info("Converting openmathinstruct ...")
    y_all = []
    for i, sample in enumerate(dataset):
        y = {
            "dataset": "openmathinstruct",
            "id": "openmathinstruct_{}".format(i),
            "messages": [{"role": "user", "content": sample["problem"]}, {"role": "assistant", "content": sample["generated_solution"]}]
        }
        y_all.append(y)

    log.info(f"In total {len(y_all)} samples")

    return ds.Dataset.from_list(y_all)

def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    
    random.seed(opts.seed)
    remove_columns = ["dataset", "id", "messages"]
    if opts.dataset == "tulu3-if":
        dataset = ds.load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
        remove_columns = ["prompt", "id", "messages", "constraints"]
    elif opts.dataset == "mmlu":
        dataset = ds.load_dataset("cais/mmlu", "auxiliary_train", split="train")
        dataset = convert_mmlu(dataset)
    elif opts.dataset == "o1-journey":
        dataset = ds.load_dataset("GAIR/o1-journey", split="train")
        dataset = convert_o1journey(dataset)
    elif opts.dataset == "openmathinstruct-2-1M":
        dataset = ds.load_dataset('nvidia/OpenMathInstruct-2', split='train_1M')
        dataset = convert_openmathinstruct(dataset)
    elif opts.dataset == "smoltalk":
        dataset = ds.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
        remove_columns = ['source', 'messages']
    else:
        raise NotImplementedError(f"The dataset {opts.dataset} is not supported.")
    
    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=remove_columns,
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Math dataset")
    parser.add_argument("--output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default=Path(__file__).parent / "tokenizers" / "allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-ds", "--dataset", type=str, help="""Dataset that we are processing.""", default="smoltalk")
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=4096)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=0)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("--seed", type=int, help="""random seed""", default=96)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)