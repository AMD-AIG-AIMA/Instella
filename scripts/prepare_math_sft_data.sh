for fold in {0..9}
do
    python scripts/prepare_sft_data.py --output_dir datasets/openmathinstruct14M_fold_$fold --tokenizer tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json --dataset openmathinstruct-2 --eos 0 --pad 1 -s 4096 -j 128 --fold $fold
done