mkdir -p /fs-computility/llm/shared/limo/data/tmp
export TMPDIR=/fs-computility/llm/shared/limo/data/tmp

python setup/download_prepare_hf_data_dclm10prct.py dclm_baseline_1.0_10prct 500 --data_dir=/fs-computility/llm/shared/limo/data/ --skip_download --nchunks=16 --output_suffix=_16chunks 