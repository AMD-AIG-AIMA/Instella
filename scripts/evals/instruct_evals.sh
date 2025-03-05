# Instruct model evals using OLMES (https://github.com/allenai/olmes/tree/main)

# Instella-3B-Instruct  (Instruct)
MODEL_NAME='amd/Instella-3B-Instruct'
RESULT_DIR='Instella-3B-Instruct'
MODEL_ARGS='{"chat_model":"True", "max_length": 4096, "device_map_option":"cuda"}'

# Llama3.2-3b-instruct  (Instruct)
MODEL_NAME='llama3.2-3b-instruct'
RESULT_DIR='llama3.2-3b-instruct'
MODEL_ARGS='{"device_map_option":"cuda"}'

# Qwen2.5-3B-Instruct  (Instruct)
MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RESULT_DIR='Qwen2.5-3B-Instruct'
MODEL_ARGS='{"chat_model":"True", "max_length": 8192,"device_map_option":"cuda"}'

# Gemma2-2b-instruct (Instruct)
MODEL_NAME='gemma2-2b-instruct'
RESULT_DIR='gemma2-2b-instruct'
MODEL_ARGS='{"device_map_option":"cuda"}'

# Stablelm-zephyr-3b (Instruct)
MODEL_NAME='stabilityai/stablelm-zephyr-3b'
RESULT_DIR='stablelm-zephyr-3b'
MODEL_ARGS='{"chat_model":"True", "max_length": 4096,"device_map_option":"cuda"}'

# OpenELM-3B-Instruct (Instruct)
MODEL_NAME='apple/OpenELM-3B-Instruct'
RESULT_DIR='OpenELM-3B-Instruct'
MODEL_ARGS='{"chat_model":"True", "max_length": 2048, "device_map_option":"cuda", "chat_template": "zephyr", "trust_remote_code": true, "tokenizer": "meta-llama/Llama-2-7b-hf", "add_bos_token": true}'

olmes --model $MODEL_NAME \
    --task 'mmlu:mc::tulu gsm8k::tulu ifeval::tulu truthfulqa::tulu gpqa:0shot_cot::llama3.1 minerva_math::llama3.1 bbh:cot-v1::tulu alpaca_eval_v2::tulu' \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/standard_benchmarks/ \
    --model-args $MODEL_ARGS


'''
For evaluating apple/OpenELM-3B-Instruct on GPQA and IFEval use the following task configs:
olmes --model apple/OpenELM-3B-Instruct \
    --task "$(cat ifeval_truncate_config.json)" \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/standard_benchmarks/ \
    --model-args '{"chat_model":"True", "max_length": 2048, "device_map_option":"cuda", "chat_template": "zephyr", "trust_remote_code": true, "tokenizer": "meta-llama/Llama-2-7b-hf", "add_bos_token": true}'

olmes --model apple/OpenELM-3B-Instruct \
    --task "$(cat gpqa_truncate_config.json)" \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/standard_benchmarks/ \
    --model-args '{"chat_model":"True", "max_length": 2048, "device_map_option":"cuda", "chat_template": "zephyr", "trust_remote_code": true, "tokenizer": "meta-llama/Llama-2-7b-hf", "add_bos_token": true}'
'''
