# python setup/download_tokenizer.py cl100k_base ./tokenizers
# export TIKTOKEN_CACHE_DIR=/fs-computility/llm/shared/llmeval/share_tiktoken
# unset HF_HUB_OFFLINE 不要在这个脚本中设定
# python setup/download_prepare_hf_data.py dclm_baseline_1.0 600 --data_dir ./data --seed 42 --nchunks 4



torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml

# python -m apps.main.train config=apps/main/configs/debug.yaml

# head -n 1 data/fineweb_edu_10bt/fineweb_edu_10bt.chunk.00000.jsonl
#  python eval_debug.py --ckpt_dir /path/to/checkpoint --dump_dir eval_results