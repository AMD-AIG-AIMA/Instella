# Pretrained model evals using OLMES (https://github.com/allenai/olmes/tree/main)


# Instella-3B  (Pre-trained)
MODEL_NAME='amd/Instella-3B'
RESULT_DIR='Instella-3B'
MODEL_ARGS='{"max_length": 4096,"device_map_option":"cuda"}'

# Gemma2-2b  (Pre-trained)
MODEL_NAME='gemma2-2b'
RESULT_DIR='gemma2'
MODEL_ARGS='{"device_map_option":"cuda"}'

# Qwen2.3-3B (Pre-trained)
MODEL_NAME='qwen2.5-3b'
RESULT_DIR='qwen2.5-3b'
MODEL_ARGS='{"max_length": 8192, "device_map_option":"cuda"}'

# Llama3.2-3b  (Pre-trained)
MODEL_NAME='llama3.2-3b'
RESULT_DIR='llama3.2-3b'
MODEL_ARGS='{"device_map_option":"cuda"}'

# Pythia-2.8b  (Pre-trained)
MODEL_NAME='EleutherAI/pythia-2.8b'
RESULT_DIR='pythia-2.8b'
MODEL_ARGS='{"max_length": 2048, "device_map_option":"cuda"}'

# OpenELM-3B  (Pre-trained)
MODEL_NAME='apple/OpenELM-3B'
RESULT_DIR='OpenELM-3B'
MODEL_ARGS='{"max_length": 2048, "trust_remote_code": "True", "device_map_option":"cuda", "trust_remote_code": true, "tokenizer": "meta-llama/Llama-2-7b-hf", "add_bos_token": true}'

# GPT-Neo-2.7B  (Pre-trained)
MODEL_NAME='EleutherAI/gpt-neo-2.7B'
RESULT_DIR='gpt-neo-2.7B'
MODEL_ARGS='{"max_length": 2048, "device_map_option":"cuda"}'


olmes --model $MODEL_NAME \
    --task 'arc_challenge::olmo1 arc_easy::olmo1 boolq::olmo1 hellaswag::olmo1 piqa::olmo1 sciq::olmo1 winogrande::olmo1 openbookqa::olmo1' \
    --batch-size "auto" \
    --output-dir ./results/$RESULT_DIR/standard_benchmarks/ \
    --model-args $MODEL_ARGS
olmes --model $MODEL_NAME \
    --task mmlu:mc::olmes \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/mmlu/ \
    --model-args $MODEL_ARGS
olmes --model $MODEL_NAME \
    --task gsm8k::olmes \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/gsm8k/ \
    --model-args $MODEL_ARGS
olmes --model $MODEL_NAME \
    --task bbh:cot-v1::olmes \
    --batch-size 8 \
    --output-dir ./results/$RESULT_DIR/bbh/ \
    --model-args $MODEL_ARGS
