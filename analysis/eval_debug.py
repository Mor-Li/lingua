# from datasets import load_dataset

# # 加载 Parquet 格式的 hellaswag 数据集
# dataset = load_dataset("hellaswag", "default", trust_remote_code=True)
# print(dataset)


from datasets import load_dataset

# 使用 Parquet 格式的分支
dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)
dataset = load_dataset("ybisk/piqa", trust_remote_code=True)
print(dataset)
# 草 原来这个bug其实不存在 这样load是能的 只是如果设定了hfendpoint就会不行
# 如果说你不用endpoint 就得proxy
# 诶 其实就算说你开了endpoint和proxy也行，只是说你得把之前的//fs-computility/mllm1/limo/.cache/huggingface/modules/datasets_modules/datasets/Rowan--hellaswag/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
# 给删了就行了

# 也有可能是我装了这个包
# 这个仓库启用了 Xet Storage（一种存储系统），但是 'hf_xet' 包没有被安装。系统将退回到常规的 HTTP 下载方式。为了获得更好的性能，建议安装该包，可以通过以下命令安装：
# pip install huggingface_hub[hf_xet] 或者
# pip install hf_xet


# from datasets import get_dataset_config_names
# print(get_dataset_config_names("hellaswag"))

