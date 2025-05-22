conda activate lingua_250403
python volc_tools.py \
    --task-cmd '. lingua_run.sh' \
    --log-level DEBUG \
    --num-gpus 8 \
    --num-replicas 1 \
    --task-name "lingua_debug" \
    --queue-name "llmeval_volc" \
    --image 'vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.4.1' \
    --yes