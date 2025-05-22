conda activate lingua_250403

# 启动32卡测试任务
python volc_tools_32gpus.py \
    --task-cmd "test_volcano_dist" \
    --log-level DEBUG \
    --num-gpus 32 \
    --num-replicas 4 \
    --task-name "volc_dist_test" \
    --queue-name "llmeval_volc" \
    --image 'vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.4.1' \
    --yes 