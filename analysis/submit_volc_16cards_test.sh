conda activate lingua_250403

# 启动16卡测试任务（2个节点 x 8卡/节点）
python volc_tools_32gpus.py \
    --task-cmd "test_volcano_dist" \
    --log-level DEBUG \
    --num-gpus 16 \
    --num-replicas 2 \
    --task-name "volc_dist_test_16cards" \
    --queue-name "llmeval_volc" \
    --image 'vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.4.1' \
    --yes 