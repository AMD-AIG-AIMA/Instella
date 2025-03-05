# Evaluation

## Requirements
Install the following packages for evaluation standard benchmark using [OLMES](https://github.com/allenai/olmes/tree/main), [FastChat MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md), and [Alpaca](https://github.com/tatsu-lab/alpaca_eval/tree/main).

```shell
git clone https://github.com/allenai/olmes.git
cd olmes
git checkout 38af8b6
pip install -e .
cd ../
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
git chekout 1ffd4a6
pip install -e ".[model_worker,llm_judge]"
cd ../
pip install git+https://github.com/tatsu-lab/alpaca_eval
export OPENAI_API_KEY=<your_api_key>
```

## Evaluation Benchmarks
Following benchmarks were used for evaluating pre-trained models:
| Benchmark | N-Shots | Primary Metric | Alias (OLMES) |
| -------- | ------- | ------- | -------|
| ARC Challenge | 0 | acc_uncond | arc_challenge::olmo1 |
| ARC Easy | 0 | acc_uncond | arc_easy::olmo1 |
| BoolQ | 0 | acc_raw | boolq::olmo1 |
| Hellaswag| 0 | acc_per_token | hellaswag::olmo1 |
| PIQA | 0  | acc_per_token | piqa::olmo1 |
| SciQ | 0  | acc_raw | sciq::olmo1 |
| Winogrande| 0 | acc_per_token | winogrande::olmo1|
| OpenBookQA | 0 | acc_uncond | openbookqa |
| MMLU | 5 | acc_raw | mmlu:mc::olmes |
| BBH | 3 | exact_match | bbh:cot-v1::olmes |
| GSM8k | 8 | exact_match| gsm8k::olmes |

Following benchmarks were used for evaluating instruct models:
| Benchmark | N-Shots | Primary Metric | Alias (OLMES) |
| -------- | ------- | ------- | ------- |
| MMLU | 5 | acc_raw | mmlu:mc::tulu |
| TruthfulQA| 6 | mc2 | truthfulqa::tulu |
| BBH | 3 | bbh:cot-v1::tulu | exact_match |
| GPQA | 0 | exact_match | gpqa:0shot_cot::llama3.1 |
| GSM8k | 8 | exact_match | gsm8k::tulu |
| Minerva MATH | 0 | exact_match_flex | minerva_math::llama3.1 |
| IFEval | 0 | prompt_level_loose_acc | ifeval::tulu |
| Alpaca | 0 | length_controlled_winrate | alpaca_eval_v2::tulu |

Use the following scripts to run benchmark evaluations using OLMES. We evaluated all the models in full-precision.
```shell
# Pre-trained Models:
source pretrain_evals.sh

# Instruct Models:
source instruct_evals.sh
```

To run MT-Bench bechmark, use the following instructions:
```shell
MODEL_PATH=<huggingface model name or path to model>
MODEL_SAVE_NAME=<model name used for saving the generations>

cd FastChat/fastchat/llm_judge/
python gen_model_answer.py --model-path $MODEL_PATH --model-id $MODEL_SAVE_NAME
python gen_judgment.py --model-list $MODEL_SAVE_NAME --parallel 2
python show_result.py \
    --model-list $MODEL_SAVE_NAME
```
In order to run MT-Bench evals on Instella models, add the following `InstellaAdapter` in `FastChat/fastchat/model/model_adapter.py`:
```python
class InstellaAdapter(BaseModelAdapter):
    """The model adapter for Instella models"""
    use_fast_tokenizer=True
    def match(self, model_path: str):
        return "instella" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("instella")
...
register_model_adapter(InstellaAdapter)
```
as well as register the Instella conversation template in `FastChat/fastchat/conversation.py`:
```python
def get_promt(self):
    ...
    elif self.sep_style == SeparatorStyle.INSTELLA:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            ret += "<|endoftext|>"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
    ...
...
register_conv_template(
    Conversation(
        name="instella",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.INSTELLA,
        sep="\n",
    )
)
```
