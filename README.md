<div align="center">
  <br>
  <br>
  <h1>Instella-Long✨: Fully Open Language Model with Long-context Capability</h1>
<a href='https://huggingface.co/amd/Instella-3B-Long-Instruct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/datasets/amd/Instella-Long'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/instella-long-context/README.html'><img src='https://img.shields.io/badge/Technical-Blog-red'></a>
</div>

Instella-Long is a long-context language model continually trained from [Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-instruct) on AMD Instinct&trade; MI300X GPUs. Instella-Long can support 128K context length and achieve competitive performance outperforming open-weights models such as Phi-3.5-mini, Gemma-3-4B, and Qwen2.5-3B on the long-context benchmark. We provide the model weights, training code, and training data to accelerate the development of open-source language models.

## Getting Started

### Installation
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. For AMD GPUs, you can aslo start with a [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch/tags?name=pytorch) docker. 

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/AMD-AIG-AIMA/Instella.git
cd Instella
git checkout instella-long
# install Flash-Attention on MI300X
GPU_ARCH=gfx942 MAX_JOBS=$(nproc) pip install git+https://github.com/Dao-AILab/flash-attention.git -v
# install other dependencies
pip install -e .[all]
```

## Training Data Download
We released all the training data at [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long). The data is in MDS format and can be loaded through [mosaicml-streaming](https://github.com/mosaicml/streaming). The data can be downloaded by cloning [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long) or the `huggingface_hub.snapshot_download` function.

## Base Model Preparation
Our long-context training starts from [amd/Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct). To start the training, we convert `amd/Instella-3B-Instruct` to an unsharded checkpoint format for the training to load from. The conversion command is:

```bash
python hf_instella/convert_hf_to_unsharded.py --hf-model amd/Instella-3B-Instruct --output-dir ./Instella-3B-Instruct-unsharded
```
The converted checkpoint is saved at `./Instella-3B-Instruct-unsharded/model.pt`, which is used to initialize the continued pre-training phase 1.


## Training 

### Continued Pre-Training 

Continued pre-training phase 1:

```bash
# single node
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-long-pretrain-phase1.yaml
# multiple nodes
torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT scripts/train.py configs/instella-3b-long-pretrain-phase1.yaml 
```

Continued pre-training phase 2:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-long-pretrain-phase2.yaml
```
 
### Supervised Fine-tuning (SFT)

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/instella-3b-long-sft.yaml
```

### Direct Preference Optimization (DPO)
We conduct DPO using [open-instruct](https://github.com/allenai/open-instruct/tree/main) with this [commit](https://github.com/allenai/open-instruct/tree/bcb991d4d9b297dc301e03ebaaa5d80dd76bb384/). Please follow their instructions to install the package. The DPO training loads the model in the huggingface format, so we first convert the SFT checkpoint to the huggingface format as follows:

```bash
python hf_instella/convert_instella_to_hf.py --checkpoint-dir outputs/instella-3b-long-sft/step250-unsharded --destination-dir outputs/instella-3b-long-sft/step250-hf --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --keep-instella-artifacts
```

Then, we run training as follows:

```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_stage2.conf \
    scripts/dpo_tune.py \
    configs/instella-3b-long-dpo.yaml
```

## Inference
An example to run inference with huggingface is illustrated as follows:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "amd/Instella-3B-Long-Instruct"

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

## Evaluation

The long-context performance is evaluated by [HELMET](https://github.com/princeton-nlp/HELMET)


## Additional Resources

Please refer to the following blogs to get started with using these techniques on AMD GPUs:

- [Introducing Instella: New State-of-the-art Fully Open 3B Language Models](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html)
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [Accelerate PyTorch Models using torch.compile on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)
- [Introducing the First AMD 1B Language Models: AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)
 
## Acknowledgement
This codebase is built from [OLMo](https://github.com/allenai/OLMo/tree/main).

## License

- The [Instella-3B-Long-Instruct](https://huggingface.co/amd/Instella-3B-Long-Instruct) model is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](./LICENSE) and [NOTICES](./NOTICES) files for more information. 
- The [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long) is a collection of pre-training and instruction following data that is used to train [Instella-3B-Long-Instruct](https://huggingface.co/amd/Instella-3B-Long-Instruct), and is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](https://huggingface.co/datasets/amd/Instella-Long/blob/main/LICENSE) in the [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long) dataset card for more information.

## Citations
Feel free to cite our Instella-3B models and give us a star⭐ if you find our work helpful :)

```text
@misc{Instella,
    title = {Instella: Fully Open Language Models with Stellar Performance},
    url = {https://huggingface.co/amd/Instella-3B},
    author = {Jiang Liu and Jialian Wu and Xiaodong Yu and Prakamya Mishra and Sudhanshu Ranjan and Zicheng Liu and Chaitanya Manem and Yusheng Su and Pratik Prabhanjan Brahma and Gowtham Ramesh and Ximeng Sun and Ze Wang and Emad Barsoum},
    month = {March},
    year = {2025}
}
```
