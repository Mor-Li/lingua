import tiktoken
import torch
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained('hf_models/model_400k')

# 直接使用tiktoken
encoding = tiktoken.get_encoding('cl100k_base')

def generate_text(prompt, max_length=100, temperature=0.7):
    """使用tiktoken生成文本，并添加attention_mask"""
    input_ids = encoding.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(model.device)
    
    # 创建attention_mask（全1表示所有token都应被注意）
    attention_mask = torch.ones_like(input_tensor)
    
    outputs = model.generate(
        input_ids=input_tensor,
        attention_mask=attention_mask,  # 显式提供attention_mask
        max_length=max_length, 
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2  # 略微增加重复惩罚
    )
    
    return encoding.decode(outputs[0].tolist())

# 使用英文prompt测试
english_prompt = "Write a short story about a robot who learns to paint."
print("===== English Input =====")
print(english_prompt)
print("\n===== Output =====")
print(generate_text(english_prompt))

# 可选：测试简单英文指令
simple_instruction = "Explain what quantum computing is in simple terms."
print("\n===== Simple Instruction =====")
print(simple_instruction)
print("\n===== Output =====")
print(generate_text(simple_instruction))

# 添加基础模型续写测试
print("\n===== Text Continuation Test 1 =====")
continuation_test1 = "The scientist looked at the experimental results with amazement. For the first time in history,"
print(continuation_test1)
print("\n===== Continuation Output =====")
print(generate_text(continuation_test1))

print("\n===== Text Continuation Test 2 =====")
continuation_test2 = "In the ancient forest, a hidden door appeared among the roots of the oldest tree. When opened,"
print(continuation_test2)
print("\n===== Continuation Output =====")
print(generate_text(continuation_test2))