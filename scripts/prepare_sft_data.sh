N_PROCESSES=200

echo "Processing smoltalk"
python scripts/prepare_sft_data.py --output_dir datasets/smoltalk --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset smoltalk --eos 0 --pad 1 -s 4096 -j $N_PROCESSES

echo "Processing openmathinstruct-2-1M"
python scripts/prepare_sft_data.py --output_dir datasets/openmathinstruct2_1M --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset openmathinstruct-2-1M --eos 0 --pad 1 -s 4096 -j $N_PROCESSES

echo "Processing tulu3-if"
python scripts/prepare_sft_data.py --output_dir datasets/tulu3-if --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset tulu3-if --eos 0 --pad 1 -s 4096 -j $N_PROCESSES

echo "Processing o1-journey"
python scripts/prepare_sft_data.py --output_dir datasets/o1-journey-10x --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset o1-journey --eos 0 --pad 1 -s 4096 -j $N_PROCESSES

echo "Processing mmlu"
python scripts/prepare_sft_data.py --output_dir datasets/mmlu --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset mmlu --eos 0 --pad 1 -s 4096 -j $N_PROCESSES