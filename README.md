<div align="center">
  <br>
  <br>
  <h1>Instella✨: Fully Open Language Models with Stellar Performance</h1>
<a href='https://huggingface.co/collections/amd/instella-67c8a2c56e9198c85a97dd08'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a> 
</div>

Instella is a family of state-of-the-art open language models trained on AMD Instinct™ MI300X GPUs by the AMD GenAI team. Instella models significantly outperform existing fully open language models of similar size, as well as bridges the gap between fully open and open weight models by achieving competitive performance compared to Llama-3.2-3B and Qwen2.5-3B models. We provide the model weights, training code, and training data to accelerate the development of open-source language models. For our vision-language model, please check out [Instella-VL](https://github.com/AMD-AIG-AIMA/InstellaVL). 

<div align="center">
<img src="figs/scaling_perf_instruct.png" style="object-fit: contain;"/>
<em><b>Figure 1:</b> Pareto frontier of pre-training tokens vs average benchmark performance for pre-trained and instruct models.</em>

[^1]

</div>

[^1]: Here even for instruct models, we compared against pre-training tokens as 1) exact open weigth instruct model training token numbers are unknown, and 2) adding instruct model training tokens (in billions) leads to marginally insignificant shift in trends.
## Getting Started

### Installation
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. For AMD GPUs, you can aslo start with a [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch/tags?name=pytorch) docker. 

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/AMD-AIG-AIMA/Instella.git
cd Instella
# install Flash-Attention on MI300X
GPU_ARCH=gfx942 MAX_JOBS=$(nproc) pip install git+https://github.com/Dao-AILab/flash-attention.git -v
# install other dependencies
pip install -e .[all]
```

### Example Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "amd/Instella-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)

prompt = [{"role": "user", "content": "What are the benefits of open-source AI research?"}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.8,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))
```

### Chat in TRL
You can also use the TRL CLI to chat with the model from the terminal:
```bash
pip install trl
trl chat --model_name_or_path amd/Instella-3B-Instruct --trust_remote_code --max_new_tokens 1024

# <root>:
# which is bigger 9.8 or 9.11?

# <amd/Instella-3B-Instruct>:
# 9.8 is bigger than 9.11. The difference between the two numbers is 0.69 (9.8 - 9.11 = 0.69), which indicates that 9.8 is 0.69 units larger than 9.11.  
```


## Pre-Training 

### Data Preparation
We use the [OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924) dataset for stage 1 pretraining. After downloading the dataset, run the following to tokenize the text data:
```bash
pip install dolma
bash scripts/prepare_pretrain_data_stage1.sh
```

To prepare the second stage training data, download the [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124), [python-edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus#downloading-the-data), [dm_math](https://huggingface.co/datasets/LLM360/TxT360/tree/main/data/dm_maths) datasets, and run the data preparation script:
```bash
bash scripts/prepare_pretrain_data_stage2.sh
```

### Training
The configs used to train the Instella-3B models are provided in the [`configs`](https://github.com/AMD-AIG-AIMA/Instella/blob/main/configs/) directory. 

Once you've updated the data paths in the config you can launch a training run via `torchrun`. For example, to launch the 3B model training on a single 8x GPU node, you would run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-pretrain-stage1.yaml
```

To resume training from a checkpoint, you can pass its path to `scripts/train.py` with the `--load_path` arguments. For example, to resume training from step 10000 of the Instella pretraining run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-pretrain-stage1.yaml --load_path output/pretrain/Instella-3B-pretrain-stage1/latest
```
To launch multi-node jobs, run the following on each of the nodes:
```bash
torchrun --nnodes=$NUM_NODES --nproc_per_node=8 --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/train.py configs/instella-3b-pretrain-stage1.yaml
```
where `NUM_NODES` is the total number of nodes, `JOB_ID` is the user-defined job id, `MASTER_ADDR` is the IP address of the master node and `MASTER_PORT` is the port on the `MASTER_ADDR` that can be used to host the C10d TCP store. Please refer to this [documentation](https://pytorch.org/docs/stable/elastic/run.html) for `torchrun` to understand the arguments to configure the rendezvous backend for multi-node training. 

For the second stage pretraining, we trained the model from the first stage checkpoints with three random seeds (see the configs: [5796](./configs/instella-3b-pretrain-stage2-seed-5796.yaml), [6198](./configs/instella-3b-pretrain-stage2-seed-6198.yaml), and [8915](./configs/instella-3b-pretrain-stage2-seed-8915.yaml)), and then merge the checkpoints with [this script](./scripts/merge_ckpts.py). 

## Supervised Fine-tuning (SFT)

### Data Preparation
Run the following commands to prepare the SFT data:
```bash
bash scripts/prepare_sft_data.sh
```
### Training 
Launch the SFT job with the [SFT config file](./configs/instella-3b-sft.yaml):

```
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-sft.yaml
```

Note: please make sure to update `load_path` to your final pretrain checkpoint.

## Direct Preference Optimization (DPO)
We conduct DPO after SFT using [open-instruct](https://github.com/allenai/open-instruct/tree/main) with this [commit](https://github.com/allenai/open-instruct/tree/bcb991d4d9b297dc301e03ebaaa5d80dd76bb384/). Please follow their instructions to install the package and then run the DPO training:

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_stage2.conf \
    scripts/dpo_tune.py \
    configs/instella-3b-dpo.yaml
```

## Evaluation

Please refer to this [folder](./scripts/evals) for detailed instructions for model evaluation.

## Generate GSM8k Synthetic Data

Synthetic data generation for GSM8k is a multi-step process:
1. Original question -> Masked question (The numerical values in the question are replaced by variables).
2. Masked question -> Program (Code to solve the masked question).
3. Program -> Perturbed questions (New questions where the values have been perturbed).
4. Perturbed questions -> Chain of thought solutions. 

Some steps are repeated multiple times until we know that the output is correct. Specifically, in steps 2 and 4, we already know the answer, so if the answer from the generated programs (or CoTs) don't match the expected answer, we re-run the previous steps. 

For steps 1 and 2, please run the following command:
```
python -W ignore scripts/generate_gsm8k_programs.py
```
  
For steps 3 and 4, please run the following command:
```
python -W ignore scripts/generate_gsm8k_new_samples.py 0
```

## Additional Resources

### Hugging Face Model Cards

- Pre-trained models:
  - Instella-3B-Stage1: [amd/Instella-3B-Stage1](https://huggingface.co/amd/Instella-3B-Stage1), First stage pre-training checkpoint.
  - Instella-3B: [amd/Instella-3B](https://huggingface.co/amd/Instella-3B), Final pre-training checkpoint.
- Instruction-tuned models:
  - Instella-3B-SFT: [amd/Instella-3B-SFT](https://huggingface.co/amd/Instella-3B-SFT), Supervised fine-tuned checkpoint.
  - Instella-3B-Instruct: [amd/Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct), Final Instruction-tuned checkpoint.

### Datasets

Second stage pre-training GSM8k synthetic dataset: [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic)

- The dataset consists of two splits: “train” and “train_119K”.
- For Instella-3B model second stage pre-training we used the “train_119K” split, which is a subset of the larger “train” split.

Please refer to the following blogs to get started with using these techniques on AMD GPUs:

- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [Accelerate PyTorch Models using torch.compile on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)
- [Introducing the First AMD 1B Language Models: AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)
 
## Acknowledgement
This codebase is built from [OLMo](https://github.com/allenai/OLMo/tree/main).

## License

- The Instella-3B models are licensed for academic and research purposes under a ResearchRAIL license. 
- The [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic) dataset used in second stage pre-training is built with Qwen2.5-72B-Instruct, and is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic/blob/main/LICENSE) and [NOTICES](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic/blob/main/NOTICES) in the [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic) dataset card files for more information.
- Refer to the [LICENSE](./LICENSE) and [NOTICES](./NOTICES) files for more information.

## Citations
Feel free to cite our Instella-3B models:

```text
@misc{Instella,
    title = {Instella: Fully Open Language Models with Stellar Performance},
    url = {https://huggingface.co/amd/Instella-3B},
    author = {Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu, Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum},
    month = {March},
    year = {2025}
}
```
