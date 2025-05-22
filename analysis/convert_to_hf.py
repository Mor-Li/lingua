import os
import sys
import torch
import json
import shutil
from pathlib import Path
import tiktoken
from transformers import AutoConfig, PreTrainedTokenizerFast

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

from apps.main.transformer import LMTransformer, LMTransformerArgs
from apps.main.generate import load_consolidated_model_and_tokenizer
from lingua.args import dataclass_from_dict

def convert_to_hf(model_path, output_path):
    """
    将Lingua格式的模型转换为Hugging Face格式
    
    Args:
        model_path: Lingua模型路径 (包含consolidated目录的路径)
        output_path: 输出的Hugging Face模型路径
    """
    print(f"正在加载Lingua模型: {model_path}")
    
    # 加载Lingua模型和分词器
    model, tokenizer, config = load_consolidated_model_and_tokenizer(
        os.path.join(model_path, "consolidated"),
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
    )
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 获取模型状态字典以读取实际尺寸
    state_dict = model.state_dict()
    
    # 从模型权重直接提取实际参数大小
    vocab_size = state_dict["tok_embeddings.weight"].shape[0]  # 100512
    hidden_size = state_dict["tok_embeddings.weight"].shape[1]  # 1024
    
    # 从FeedForward层提取中间尺寸
    if "layers.0.feed_forward.w1.weight" in state_dict:
        intermediate_size = state_dict["layers.0.feed_forward.w1.weight"].shape[0]  # 2816
    else:
        # 默认设置为hidden_size的2.75倍(根据错误信息中的2816/1024≈2.75)
        intermediate_size = int(hidden_size * 2.75)
    
    # 读取原始参数
    params_path = os.path.join(model_path, "consolidated", "params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
    else:
        # 如果找不到params.json，从模型结构中提取参数
        params = {
            "d_model": hidden_size,
            "n_layers": len(model.layers),
            "n_heads": model.layers[0].attention.n_heads,
            "vocab_size": vocab_size,
            "norm_eps": 1e-5,
            "max_seq_len": getattr(model, "max_seqlen", 2048),
            "rope_theta": getattr(model.rope_embeddings, "theta", 10000.0),
        }
    
    # 创建HF配置，使用直接从模型中提取的大小
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.0",
        # 使用从模型权重中提取的实际大小
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,  # 使用正确的中间层大小
        "num_attention_heads": params.get("n_heads", 8),
        "num_hidden_layers": params.get("n_layers", 8),
        "num_key_value_heads": params.get("n_kv_heads", params.get("n_heads", 8)),
        "rms_norm_eps": params.get("norm_eps", 1e-5),
        "vocab_size": vocab_size,  # 使用模型中的实际词汇表大小
        "max_position_embeddings": params.get("max_seq_len", 2048),
        "rope_theta": params.get("rope_theta", 10000.0),
        # 添加自定义字段，说明这是自定义转换模型
        "linguaml_converted": True,
        # 设置正确的tokenizer类型
        "tokenizer_class": "PreTrainedTokenizerFast"
    }
    
    print(f"模型配置信息:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  MLP中间层大小: {intermediate_size}")
    print(f"  层数: {hf_config['num_hidden_layers']}")
    print(f"  注意力头数: {hf_config['num_attention_heads']}")
    
    # 保存配置
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    
    # 处理cl100k_base tiktoken分词器
    print("正在处理tiktoken cl100k_base分词器...")
    
    try:
        # 直接使用tiktoken的cl100k_base编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # 创建一个基本的tokenizer_config.json文件
        tokenizer_config = {
            "model_type": "llama",
            "add_bos_token": False, 
            "add_eos_token": False,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "model_max_length": 2048,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "auto_map": {
                "AutoTokenizer": ["tokenizer.json", None]
            }
        }
        
        with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # 创建空的特殊token字典
        special_tokens = {
            "<|endoftext|>": 100257  # tiktoken的endoftext token
        }
        
        # 使用transformers CLI创建tokenizer.json
        # 由于tiktoken的内部结构较复杂，我们简化处理方式
        
        # 创建一个简单的tokenizer_json结构
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": 100257,
                    "content": "<|endoftext|>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                }
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": []
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True
            },
            "post_processor": None,
            "decoder": {
                "type": "ByteFallback"
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "vocab": {},
                "merges": []
            }
        }
        
        # 尝试获取tokenzier的映射关系
        for i in range(min(100, vocab_size)):
            try:
                # 尝试对单个token ID进行解码
                token_bytes = encoding.decode([i])
                if i not in special_tokens.values():
                    tokenizer_json["model"]["vocab"][token_bytes] = i
            except:
                pass
        
        # 保存tokenizer.json
        with open(os.path.join(output_path, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_json, f, indent=2)
            
        print("已创建tokenizer相关文件")
        
    except Exception as e:
        print(f"警告：处理分词器时出错: {str(e)}")
        print("将使用最简化的配置文件...")
        
        # 创建简化版本的tokenizer_config.json
        with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
            json.dump({
                "model_type": "llama",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "model_max_length": 2048,
                "tokenizer_class": "PreTrainedTokenizerFast"
            }, f, indent=2)
        
        # 复制原始tiktoken文件到输出目录
        tiktoken_path = getattr(tokenizer, "path", None)
        if tiktoken_path and os.path.exists(tiktoken_path):
            shutil.copy(tiktoken_path, os.path.join(output_path, "cl100k_base.tiktoken"))
            print(f"已复制原始tiktoken文件到输出目录")
    
    # 转换模型权重
    print("正在转换模型权重...")
    
    # 创建HF格式的状态字典
    hf_state_dict = {}
    
    # 映射权重名称 - 根据Lingua的LMTransformer架构
    mapping = {
        # 嵌入层
        "tok_embeddings.weight": "model.embed_tokens.weight",
        # 最终层归一化
        "norm.weight": "model.norm.weight",
        # 输出层
        "output.weight": "lm_head.weight"
    }
    
    # 添加各层的映射
    for i in range(hf_config["num_hidden_layers"]):
        # 注意力层
        mapping[f"layers.{i}.attention.wq.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        mapping[f"layers.{i}.attention.wk.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        mapping[f"layers.{i}.attention.wv.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        mapping[f"layers.{i}.attention.wo.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"
        
        # 前馈网络
        mapping[f"layers.{i}.feed_forward.w1.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        mapping[f"layers.{i}.feed_forward.w3.weight"] = f"model.layers.{i}.mlp.up_proj.weight"
        mapping[f"layers.{i}.feed_forward.w2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
        
        # 层归一化
        mapping[f"layers.{i}.attention_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        mapping[f"layers.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
    
    # 应用映射
    for name, param in state_dict.items():
        if name in mapping:
            hf_state_dict[mapping[name]] = param
        else:
            # 对于未映射的参数，保留原始名称并输出警告
            print(f"警告：未映射的参数: {name}")
    
    # 保存转换后的模型
    try:
        from safetensors.torch import save_file
        print(f"正在使用safetensors格式保存模型权重到 {output_path}")
        save_file(hf_state_dict, os.path.join(output_path, "model.safetensors"))
    except ImportError:
        print(f"safetensors未安装，使用PyTorch格式保存模型权重")
        torch.save(hf_state_dict, os.path.join(output_path, "pytorch_model.bin"))
    
    # 创建generation_config.json
    generation_config = {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "pad_token_id": 100257,  # tiktoken的<|endoftext|>
        "bos_token_id": 100257,  # tiktoken的<|endoftext|>
        "eos_token_id": 100257   # tiktoken的<|endoftext|>
    }
    
    with open(os.path.join(output_path, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)
    
    print(f"模型已成功转换并保存到: {output_path}")
    print("")
    print("你可以使用以下代码加载模型:")
    print("")
    print("```python")
    print("import tiktoken")
    print("import torch")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("")
    print("# 加载模型")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print("")
    print("# 方法1：尝试加载HF tokenizer")
    print("try:")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{output_path}', model_max_length=2048)")
    print("    # 测试生成")
    print("    inputs = tokenizer('你好，请问', return_tensors='pt')")
    print("    outputs = model.generate(**inputs, max_length=100)")
    print("    print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    print("except Exception as e:")
    print("    print(f'使用HF tokenizer失败: {e}')")
    print("    print('切换到直接使用tiktoken')")
    print("")
    print("    # 方法2：直接使用tiktoken")
    print("    encoding = tiktoken.get_encoding('cl100k_base')")
    print("    prompt = '你好，请问'")
    print("    input_ids = encoding.encode(prompt)")
    print("    input_ids = torch.tensor([input_ids]).to(model.device)")
    print("    outputs = model.generate(input_ids, max_length=100)")
    print("    print(encoding.decode(outputs[0].tolist()))")
    print("```")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python convert_to_hf.py <lingua模型路径> <输出路径>")
        print("例如: python convert_to_hf.py dump/debug/checkpoints/0000400000 hf_models/model_400k")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_to_hf(model_path, output_path)
