MAX_SIZE='2_147_483_648'
N_PROCESSES=200
INPUT_PATH='datasets/OLMoE-mix-0924'
OUTPUT_PATH='datasets/OLMoE-mix-0924-tokenized'

data="open-web-math"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.jsonl.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="algebraic-stack"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="arxiv"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="pes2o"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="starcoder"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="wiki"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.gz \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

data="dclm"
echo Processing $data
dolma tokens \
--documents $INPUT_PATH/data/$data/*.json.zst \
--destination $OUTPUT_PATH/$data \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES