# Meta Lingua

**Mathurin Videau***, **Badr Youbi Idrissi***, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, David Lopez-Paz. ***Equal and main contribution**

Meta Lingua is a minimal and fast LLM training and inference library designed for research. Meta Lingua uses easy-to-modify PyTorch components in order to try new architectures, losses, data, etc. We aim for this code to enable end to end training, inference and evaluation as well as provide tools to better understand speed and stability. While Meta Lingua is currently under development, we provide you with multiple `apps` to showcase how to use this codebase.

<p align="center">  
 <img src="lingua_overview.svg" width="100%"/>
</p>

## Quick start

The following commands launch a SLURM job that creates an environment for Meta Lingua.
The env creation should take around 5 minutes without counting downloads. 

```bash
git clone https://github.com/facebookresearch/lingua
cd lingua

bash setup/create_env.sh
# or if you have access to a SLURM cluster
sbatch setup/create_env.sh
```
Once that is done your can activate the environment 
```bash
conda activate lingua_<date>
```
use the provided script to download and prepare data from huggingface (among `fineweb_edu`, `fineweb_edu_10bt`, or `dclm_baseline_1.0`).
This command will download the `fineweb_edu` and prepare it for training in the `./data` directory, specifying the amount of memory `terashuf` (the tool used to shuffle samples) will be allocated. By default, the number of chunks (`nchunks`) is 32. If you are running on fewer than 32 GPUs, it is recommended to set `nchunks` to 1 or to match `nchunks` with the number of GPUs (`nchunks` = NGPUs). See [here](https://github.com/facebookresearch/lingua/issues/55#issuecomment-2483643076) for more details.
```bash
python setup/download_prepare_hf_data.py fineweb_edu <MEMORY> --data_dir ./data --seed 42 --nchunks <NCHUNKS>
```
to download tokenizer (here llama3), use the folowing script:
```bash
python setup/download_tokenizer.py llama3 <SAVE_PATH> --api_key <HUGGINGFACE_TOKEN>
```
Now launch a debug job to check if everything works.  **The provided configurations are templates, you need to adapt them for them to work (change `dump_dir`, `data.root_dir`, `data.tokenizer.path`, etc ...)**

```bash
# stool stands for SLURM tool !
python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition>
# if you want to launch locally you can use torchrun
torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml
# or you can also launch on 1 GPU
python -m apps.main.train config=apps/main/configs/debug.yaml
```

When using `stool`, if a job crashes, it can be relaunched using sbatch:
```bash
sbatch path/to/dump_dir/submit.slurm
```
## 训练结果

我们在许多下游任务中获得了非常强的性能，并匹配了[DCLM baseline 1.0](https://arxiv.org/abs/2406.11794)的性能。

### 1B模型在60B DCLM tokens上的表现
| name           | arc_challenge | arc_easy | boolq |  copa | hellaswag |  obqa |  piqa |  siqa | winogrande |  nq  |  tqa  |
|----------------|:-------------:|:--------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:----------:|:----:|:-----:|
| Transformer 1B |     36.48     |   62.83  | 62.57 | 79.00 |   63.62   | 37.40 | 75.14 | 45.19 |    61.64   | 8.75 | 26.31 |
| minGRU 1B      |     30.82     |   57.89  | 62.05 | 74.00 |   50.27   | 37.00 | 72.31 | 43.76 |    52.49   | 3.24 |  9.03 |
| minLSTM 1B     |     31.76     |   60.04  | 62.02 | 73.00 |   53.39   | 36.40 | 72.36 | 45.09 |    52.80   | 4.52 | 12.73 |
| Hawk 1B        |     34.94     |   63.68  | 62.42 | 76.00 |   63.10   | 38.20 | 73.23 | 46.01 |    55.33   | 8.42 | 23.58 |
| Mamba 1B       |     35.54     |   63.42  | 62.63 | 74.00 |   64.16   | 38.80 | 75.24 | 45.14 |    60.14   | 8.84 | 26.64 |

### 7B models

| name                             | arc_challenge | arc_easy | boolq | copa  | hellaswag | obqa  | piqa  | siqa  | winogrande | mmlu  | nq    | tqa   | bbh   |
|----------------------------------|---------------|----------|-------|-------|-----------|-------|-------|-------|------------|-------|-------|-------|-------|
| Mamba 7B 200B tokens             | 47.21         | 76.03    | 65.63 | 84.00 | 77.80     | 44.00 | 80.25 | 49.69 | 70.24      | 32.81 | 20.53 | 51.93 | 20.35 |
| Llama 7B 200B tokens             | 46.95         | 75.73    | 64.80 | 84.00 | 77.45     | 45.00 | 80.20 | 48.26 | 70.32      | 48.64 | 20.66 | 51.01 | 31.47 |
| Llama 7B squared relu 1T tokens  | 49.61         | 76.74    | 72.45 | 89.00 | 81.19     | 44.80 | 82.05 | 49.95 | 72.14      | 60.56 | 25.68 | 59.52 | 42.11 |

## Project overview

Meta Lingua is structured as follows:

```
📦meta-lingua
 ┣ 📂lingua # Core library
 ┃ ┣ 📜args.py
 ┃ ┣ 📜checkpoint.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜distributed.py
 ┃ ┣ 📜float8.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optim.py
 ┃ ┣ 📜probe.py
 ┃ ┣ 📜profiling.py
 ┃ ┣ 📜stool.py
 ┃ ┣ 📜tokenizer.py
 ┃ ┗ 📜transformer.py
 ┣ 📂setup
 ┃ ┣ 📜create_env.sh
 ┃ ┗ 📜download_prepare_hf_data.py
 ┗ 📂apps # Apps that put components together
   ┣ 📂main # Main language modeling app with llama
   ┃ ┣ 📂configs
   ┃ ┣ 📜eval.py
   ┃ ┣ 📜generate.py
   ┃ ┣ 📜train.py
   ┃ ┗ 📜transformer.py
   ┣ 📂fastRNN 
   ┃ ┣ 📂component
   ┃ ┣ 📂hawk
   ┃ ┣ 📂minGRU
   ┃ ┣ 📂minLSTM
   ┣ 📂mamba
   ┣ 📂mtp # Multi token prediction
   ┗ 📂plots
```

`lingua`文件夹包含一些基本且可重用的组件，而`apps`文件夹包含将这些组件组合在一起的脚本。例如，主要的训练循环位于`apps/main`中。我们强烈建议您将其作为模板，并根据您的实验需求随意修改。

在Meta Lingua中没有什么是不可改变的。我们特意尝试使其尽可能易于修改！所以请随意分支并修改任何内容。

以下是最重要文件和功能的简要描述：

- **`transformer.py`**：定义模型架构。这是纯PyTorch `nn.Module`！这里没有什么花哨的东西。
- **`distributed.py`**：处理在多个GPU上分布模型。这是通过`parallelize_module`函数完成的，该函数包装您的普通`nn.Module`并应用几乎任何数据并行、完全分片数据并行、模型并行、`torch.compile`、激活检查点和`float8`的组合。
- **`data.py`**：LLM预训练的数据加载器。

<p align="center">  
 <img src="dataloader.png" width="40%"/>
</p>

- **`profiling.py`**：xformers分析器的小包装器，提供自动MFU和HFU计算，并在转储目录的分析文件夹中转储分析跟踪。它还具有内存分析跟踪。
- **`checkpoint.py`**：管理模型检查点。它将模型保存在转储目录的checkpoints文件夹中，采用.distcp格式，这是新的PyTorch分布式保存方法。此格式允许使用不同数量的GPU和不同的分片重新加载模型。您还可以使用`torch.distributed.checkpoint.format_utils.dcp_to_torch_save`将它们转换为普通PyTorch检查点，反之亦然，使用`torch_save_to_dcp`。
- **`args.py`**：用于处理配置的实用工具。

## 配置

大多数组件需要配置，我们选择使用数据类来表示这些配置对象。`args.py`帮助在`config.yaml`和配置字典之间进行转换，分别转换为各自的数据类。

例如，`apps/main/train.py`中的`TrainArgs`具有`LMTransformerArgs`、`OptimArgs`等子类。

以下是一个将转换为`TrainArgs`的示例配置文件：

```yaml
# 这是Meta Lingua将存储与实验相关的任何内容的位置。
dump_dir: /path/to/dumpdir
name: "debug"
steps: 1000

seed: 12

optim:
    lr: 3e-4
    warmup: 2000
    lr_min_ratio: 0.000001
    clip: 10.0

distributed:
    fsdp_type: full_shard
    compile: true
    selective_activation_checkpointing: false

model:
    dim: 1024
    n_layers: 8
    n_heads: 8

data:
    root_dir: data/shuffled
    sources:
      wikipedia: 80.0
      arxiv: 20.0
    batch_size: 32
    seq_len: 1024
    load_async: true
    tokenizer:
        name: sp
        path: tokenizers/llama2.model
```


## Launching jobs

### Command line arguments

所有脚本（`train.py`、`eval.py`、`stool.py`）中的命令行接口使用[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments)
它接受点列表形式的参数
如果数据类如下所示
```python
@dataclass
class DummyArgs:
    name: str = "blipbloup"
    mode: LMTransformerArgs = LMTransformerArgs()
    
@dataclass
class LMTransformerArgs:
    dim: int = 512
    n_layers: int = 12
```

Then you can pass `model.dim = 32` to change values in `LMTransformerArgs`
or just `name = tictac` for top level attributes.

**`train.py`** 简单地接受作为参数的配置文件路径，并加载该配置。行为如下：
1. 我们用默认值实例化`TrainArgs`
2. 我们用提供的配置文件中的值覆盖那些默认值
3. 我们用通过命令行提供的额外参数覆盖结果

If we take the `DummyArgs` example above, calling `train.py` with `train.py config=debug.yaml model.dim=64 name=tictac` 
where `debug.yaml` contains 
```yaml
model:
    n_layers: 24
```
将启动训练，使用配置
```python
DummyArgs(name="tictac", LMTransformerArgs(dim=64, n_layers=24))
```

### 使用SLURM启动

由于我们想要进行分布式训练，我们需要`train.py`运行N次（N是GPU的数量）

最简单的方法是通过SLURM。为了简化这一点，我们提供了`lingua/stool.py`，这是一个简单的python脚本，
1. 将提供的配置保存到`dump_dir`
2. 将当前代码复制到`dump_dir`以备份
3. 创建一个sbatch文件`submit.slurm`，然后使用提供的配置启动任务。

可以通过命令行使用它

```bash
python -m lingua.stool config=apps/main/configs/debug.yaml nodes=1 account=fair_amaia_cw_codegen qos=lowest
```

或者直接使用`launch_job`函数。这允许您例如在jupyter notebook中创建许多任意配置（用于参数扫描、进行消融），并直接从那里启动任务。

由于配置文件被复制到`dump_dir`，一个简单的迭代方法是简单地更改配置文件并重新启动相同的命令。

## 调试
为了快速迭代，最好不必每次都等待SLURM分配。您可以改为请求SLURM为您分配资源，一旦分配完成，您可以在同一分配上运行多个命令。

例如您可以这样做：

```bash
salloc --nodes 2 --cpus-per-gpu 16 --mem 1760GB --gres=gpu:8 --exclusive --time=72:00:00
```

这将为您在当前终端中提供2个节点。一旦分配完成，您将看到一些自动添加的SLURM环境变量，例如`$SLURM_JOB_ID`等。这允许您例如在同一终端中执行以下命令：

```bash
srun -n 16 python -m apps.main.train config=apps/main/configs/debug.yaml
```

这将运行`python -m apps.main.train config=apps/main/configs/debug.yaml`命令在每个16个GPU上。如果这个崩溃或结束，您可以简单地重新启动`srun`，因为节点已经分配给您，您不必等待SLURM再次为您提供资源。

This will also show you the outputs of all those commands in the same terminal which might become cumbersome. 

Instead you can use `stool` directly to configure logs to be separated into different files per GPU.

```bash
python -m lingua.stool config=apps/main/configs/debug.yaml nodes=2 launcher=bash dirs_exists_ok=true
```

请注意，我们添加了 **`launcher=bash`**，这基本上意味着生成的 `submit.slurm` 将直接执行，而不是通过 `sbatch` 提交。`submit.slurm` 中也有一个 `srun` 命令，所以这与上面的 `srun` 命令非常相似。我们还添加了 **`dirs_exists_ok=true`** 来告诉 `stool` 可以覆盖现有文件夹中的内容（代码、配置等）。

如果您想使用 `pdb` 逐步调试代码，应该使用 `-n 1` 只在1个GPU上运行。

## Evaluations

Evaluations can run either during training periodically or you directly launch evals on a given checkpoint as follows:

```bash
srun -n 8 python -u -m apps.main.eval config=apps/main/configs/eval.yaml
```

You need to specify the checkpoint and dump dir of the evaluation in that config

Or through `stool` with

```bash
python -m lingua.stool script=apps.main.eval config=apps/main/configs/eval.yaml nodes=1 account=fair_amaia_cw_codegen qos=lowest
```

## Dump dir structure

```
📂example_dump_dir
 ┣ 📂checkpoints
 ┃ ┣ 📂0000001000
 ┃ ┣ 📂0000002000
 ┃ ┣ 📂0000003000
 ┃ ┣ 📂0000004000
 ┃ ┣ 📂0000005000
 ┃ ┣ 📂0000006000
 ┃ ┣ 📂0000007000 # Checkpoint and train state saved every 1000 steps here
 ┃ ┃ ┣ 📜.metadata
 ┃ ┃ ┣ 📜__0_0.distcp
 ┃ ┃ ┣ 📜__1_0.distcp
 ┃ ┃ ┣ 📜params.json
 ┃ ┃ ┣ 📜train_state_00000.json
 ┃ ┃ ┗ 📜train_state_00001.json
 ┣ 📂code # Backup of the code at the moment the job was launched
 ┣ 📂logs
 ┃ ┗ 📂166172 # Logs for each GPU in this SLURM job.
 ┃ ┃ ┣ 📜166172.stderr
 ┃ ┃ ┣ 📜166172.stdout
 ┃ ┃ ┣ 📜166172_0.err
 ┃ ┃ ┣ 📜166172_0.out
 ┃ ┃ ┣ 📜166172_1.err
 ┃ ┃ ┗ 📜166172_1.out
 ┣ 📂profiling
 ┃ ┣ 📂memory_trace_plot # Trace of memory usage through time for all GPUs
 ┃ ┃ ┣ 📜000102_h100-192-145_451082.html
 ┃ ┃ ┣ 📜000102_h100-192-145_451083.html
 ┃ ┗ 📂profile_CPU_CUDA_000104 # Profiling traces for all GPUs
 ┃ ┃ ┣ 📜h100-192-145_451082.1720183858874741723.pt.trace.json.gz
 ┃ ┃ ┗ 📜h100-192-145_451083.1720183858865656716.pt.trace.json.gz
 ┣ 📜base_config.yaml
 ┣ 📜config.yaml
 ┣ 📜metrics.jsonl
 ┗ 📜submit.slurm
```

## Related repositories

在这里，我们强调一些与本项目互补的相关工作。其中最重要的是[torchtitan](https://github.com/pytorch/torchtitan)和[torchtune](https://github.com/pytorch/torchtune)。

Lingua专为那些想要尝试LLM预训练新想法并快速获得训练/推理速度和下游基准测试反馈的研究人员设计。我们的目标是通过提供轻量级和专注的代码库，降低LLM研究的入门门槛。

我们将torchtitan、torchtune和lingua视为互补工具。Torchtitan非常适合大规模工作，因为它具有3D并行性，并且由于与PyTorch团队的紧密联系，可能会更快地集成最新的PyTorch分布式训练功能。另一方面，Torchtune在微调方面表现出色，特别是在GPU资源有限的情况下，它提供了各种微调策略，如LoRA、QLoRA、DPO和PPO。

一个典型的工作流程可能是这样的：你可能首先在Lingua中测试新想法，然后使用Torchtitan进一步扩展，最后使用Torchtune进行指令或偏好微调。

虽然这些代码库之间肯定有一些重叠，但我们认为为LLM工作的不同方面提供专注的工具是有价值的。例如，Torchtitan旨在以干净、简洁的代码库展示PyTorch最新的分布式训练功能，但对于大多数研究来说，你真的不需要PyTorch提供的每一个功能，或者在4096个GPU上扩展到100B参数的能力。例如，我们认为FSDP + torch compile将满足研究人员90%的所有需求。使用lingua，我们尝试问："得出关于想法X可扩展性的可靠结论所需的最小功能集是什么？"

我们相信这种有针对性的方法可以帮助研究人员更快地取得进展，而不必承担使用许多可能不需要的技术的心理负担。

## Citation

```
@misc{meta_lingua,
  author = {Mathurin Videau, Badr Youbi Idrissi, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, David Lopez-Paz},
  title = {{Meta Lingua}: A minimal {PyTorch LLM} training library},
  url = {https://github.com/facebookresearch/lingua},
  year = {2024}
}
```
## License

Meta Lingua 使用 BSD-3-Clause 许可证。请参阅顶级目录中的 LICENSE 文件。
