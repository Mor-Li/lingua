# 简单测试脚本用于评测模型

import os
import json
import logging
import torch
from pathlib import Path
from dataclasses import asdict

from apps.main.eval import (
    EvalHarnessLM,
    LMHarnessArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.generate import PackedCausalTransformerGenerator, PackedCausalTransformerGeneratorArgs
from lingua.distributed import setup_torch_distributed, DistributedArgs, get_global_rank
from lm_eval import simple_evaluate

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_eval():
    # 初始化分布式环境
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    
    # 模型路径
    consolidate_path = "dump/debug/checkpoints/0000005000/consolidated"
    
    # 确保路径存在
    if not Path(consolidate_path).exists():
        raise FileNotFoundError(f"路径不存在: {consolidate_path}")
    
    # 加载模型
    logger.info("正在加载模型...")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
    )
    logger.info("模型加载完成")
    
    # 设置为评估模式
    model.eval()
    
    # 创建生成器
    generator_args = PackedCausalTransformerGeneratorArgs()
    generator = PackedCausalTransformerGenerator(generator_args, model, tokenizer)
    
    # 创建评测包装器
    wrap = EvalHarnessLM(generator)
    
    # 设置评测参数
    harness_args = LMHarnessArgs(
        tasks=['hellaswag', 
               'piqa',
               'arc_challenge',
               'arc_easy',
               'boolq',
               'cb',
               'copa',
               'ybisk/piqa'
               ],  # 简化任务列表以加快测试
        num_fewshot=0,  # 设置为0以加快测试
        limit=10,  # 限制每个任务的样本数量以加快测试
    )
    
    # 运行评测
    logger.info(f"开始评测，参数: {harness_args}")
    results = simple_evaluate(wrap, **asdict(harness_args))
    
    # 输出结果
    if get_global_rank() == 0:
        logger.info(f"评测结果: {results['results']}")
        
        # 保存结果到文件
        output_dir = "test_eval_results"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"结果已保存到 {output_dir}/results.json")

if __name__ == "__main__":
    test_simple_eval()