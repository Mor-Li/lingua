import os
import sys
import torch
import gradio as gr
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.generate import (
    PackedCausalTransformerGenerator, 
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer
)
from lingua.args import dataclass_from_dict
from omegaconf import OmegaConf

# 设置多个模型路径
MODEL_PATHS = {
    "50K步": "dump/debug/checkpoints/0000050000/consolidated",
    "155K步": "dump/debug/checkpoints/0000155000/consolidated",
    "250K步": "dump/debug/checkpoints/0000250000/consolidated",
    "400K步": "dump/debug/checkpoints/0000400000/consolidated"
}

def load_models():
    """加载多个模型和分词器"""
    generators = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        print(f"正在加载模型 {model_name}...")
        try:
            model, tokenizer, config = load_consolidated_model_and_tokenizer(
                model_path,
                model_cls=LMTransformer,
                model_args_cls=LMTransformerArgs,
            )
            
            # 创建生成器配置
            gen_cfg = PackedCausalTransformerGeneratorArgs(
                temperature=0.7,  # 可以调整温度
                top_p=0.95,       # 使用top_p采样
                max_gen_len=512,  # 最大生成长度
                dtype="bf16",     # 使用BF16精度
                device="cuda",    # 使用CUDA设备
                show_progress=True
            )
            
            # 创建生成器
            generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
            generators[model_name] = generator
            print(f"模型 {model_name} 加载完成")
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {str(e)}")
    
    return generators

def generate_text_with_all_models(prompt, max_length=64, temperature=0.7, top_p=0.95):
    """使用所有模型生成文本"""
    results = {}
    
    for model_name, generator in generators.items():
        print(f"使用模型 {model_name} 生成文本...")
        # 更新生成器配置
        generator.temperature = temperature
        generator.top_p = top_p
        generator.max_gen_len = max_length
        
        # 生成文本
        generations, _, _ = generator.generate([prompt])
        results[model_name] = generations[0]
    
    return results

# 加载所有模型
print("正在初始化模型...")
generators = load_models()
print("所有模型初始化完成")

# 创建Gradio界面
def gradio_interface(prompt, max_length=256, temperature=0.7, top_p=0.95):
    results = generate_text_with_all_models(prompt, max_length, temperature, top_p)
    
    # 返回每个模型的输出结果（按照MODEL_PATHS中的顺序）
    return [results["50K步"], results["155K步"], results["250K步"], results["400K步"]]

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=5, placeholder="请输入提示文本...", label="输入提示"),
        gr.Slider(minimum=16, maximum=1024, value=256, step=16, label="最大生成长度"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="温度 (Temperature)"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p 采样")
    ],
    outputs=[
        gr.Textbox(lines=5, label="50K步模型输出"),
        gr.Textbox(lines=5, label="155K步模型输出"),
        gr.Textbox(lines=5, label="250K步模型输出"),
        gr.Textbox(lines=5, label="400K步模型输出"),
    ],
    title="多模型文本生成对比演示",
    description="输入提示文本，多个不同训练步数的模型将同时为你生成后续内容，帮助你理解模型在不同训练阶段的能力差异。"
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True 可以生成一个公共链接，方便在其他设备上访问 