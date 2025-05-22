conda activate lingua_250403


# 启动32卡训练任务
python volc_tools_32gpus.py \
    --task-cmd "apps.main.train config=apps/main/configs/debug.yaml" \
    --log-level DEBUG \
    --num-gpus 32 \
    --num-replicas 4 \
    --task-name "lingua_32gpu_training" \
    --queue-name "llmeval_volc" \
    --image 'vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.4.1' \
    --yes