#### Tokenize Domino-mix-1124
MAX_SIZE='1_147_483_648'
N_PROCESSES=200
INPUT_PATH='datasets/dolmino-mix-1124/data'
OUTPUT_PATH='datasets/dolmino-mix-1124-tokenized'

# flan
# we randomly select 50% of flan data for training
dolma tokens \
--documents $INPUT_PATH/flan/*.json.gz \
--destination $OUTPUT_PATH/flan \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

# stackexchange
dolma tokens \
--documents $INPUT_PATH/stackexchange/*.json.gz \
--destination $OUTPUT_PATH/stackexchange \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES


# tulu_math
dolma tokens \
--documents $INPUT_PATH/math/tulu_math/personahub_interm_alg/*.jsonl $INPUT_PATH/math/tulu_math/personahub_math_grade/*.jsonl $INPUT_PATH/math/tulu_math/personahub_math_v5/*.jsonl \
--destination $OUTPUT_PATH/math/tulu_math \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES


# tinyGSM-MIND
dolma tokens \
--documents $INPUT_PATH/math/tinyGSM-MIND/2students/*.jsonl.gz $INPUT_PATH/math/tinyGSM-MIND/problem-solving/*.jsonl.gz \
--destination $OUTPUT_PATH/math/tinyGSM-MIND \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes 20

# metamath
dolma tokens \
--documents $INPUT_PATH/math/metamath-owmfilter/*.jsonl.gz \
--destination $OUTPUT_PATH/math/metamath \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

# mathcoder2
dolma tokens \
--documents $INPUT_PATH/math/mathcoder2-synthmath/m-a-p_Matrix/filtered-math/*.jsonl $INPUT_PATH/math/mathcoder2-synthmath/ajibawa-2023/*.jsonl \
--destination $OUTPUT_PATH/math/mathcoder2 \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

#gsm8k
dolma tokens \
--documents $INPUT_PATH/math/gsm8k/main/train/*.jsonl.zst $INPUT_PATH/math/gsm8k/socratic/train/*.jsonl.zst \
--destination $OUTPUT_PATH/math/gsm8k \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

#dolmino_math_synth
dolma tokens \
--documents $INPUT_PATH/math/dolmino_math_synth/basic_math/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm8k-synth/resample_v1_6x/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/debate/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/interview/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/layman-knowall/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/professors/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/students/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/teacher-student/*.jsonl $INPUT_PATH/math/dolmino_math_synth/gsm_mind/probem-solving/*.jsonl \
--destination $OUTPUT_PATH/math/dolmino_math_synth \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES


#codesearchnet-owmfilter
dolma tokens \
--documents $INPUT_PATH/math/codesearchnet-owmfilter/go/*.jsonl.gz $INPUT_PATH/math/codesearchnet-owmfilter/java/*.jsonl.gz $INPUT_PATH/math/codesearchnet-owmfilter/javascript/*.jsonl.gz $INPUT_PATH/math/codesearchnet-owmfilter/php/*.jsonl.gz $INPUT_PATH/math/codesearchnet-owmfilter/python/*.jsonl.gz $INPUT_PATH/math/codesearchnet-owmfilter/ruby/*.jsonl.gz \
--destination $OUTPUT_PATH/math/codesearchnet \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

# dclm
# we randomly select 3.23% of dclm data for training
dolma tokens \
--documents $INPUT_PATH/dclm/*/*.json.zst \
--destination $OUTPUT_PATH/dclm \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

# wiki
dolma tokens \
--documents $INPUT_PATH/wiki/*.json.gz \
--destination $OUTPUT_PATH/wiki \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

# pes2o
# we randomly select 5.3% of flan data for training
dolma tokens \
--documents $INPUT_PATH/pes2o/*.json.gz \
--destination $OUTPUT_PATH/pes2o \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes 220

#### Tokenize python edu
# we randomly select 50% of python-edu data for training
dolma tokens \
--documents datasets/python-edu/*.jsonl.gz \
--destination datasets/python-edu-tokenized \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size $MAX_SIZE \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes $N_PROCESSES

#### Tokenize dm_maths
dolma tokens \
--documents datasets/dm_maths/full_data_1/*.jsonl.gz datasets/dm_maths/full_data_2/*.jsonl.gz \
--destination datasets/dm_maths-tokenized \
--tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
--max_size '1_147_483_648' \
--seed 0 \
--tokenizer.eos_token_id 0 \
--tokenizer.pad_token_id 1 \
--processes 200


#### Convert SFT datasets to pretrain format and tokenize the data
for dataset in tulu-3 code-feedback OpenHermes WebInstructSub ultrachat-200k instella-gsm8k-synthetic
do
    python scripts/convert_sft_data.py --dataset $dataset --output_dir datasets/$dataset
    
    dolma tokens \
    --documents datasets/$dataset/*.jsonl.gz \
    --destination datasets/sft/$dataset \
    --tokenizer.name_or_path tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
    --max_size $MAX_SIZE \
    --seed 0 \
    --tokenizer.eos_token_id 0 \
    --tokenizer.pad_token_id 1 \
    --processes $N_PROCESSES
done

