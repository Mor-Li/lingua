import os  # 导入操作系统模块
import subprocess  # 导入子进程模块
import uuid  # 导入UUID模块
import yaml  # 导入YAML模块
import argparse  # 导入命令行参数解析模块
from typing import Dict, Optional, List  # 导入类型提示
from dataclasses import dataclass, field, asdict  # 导入数据类相关模块
from loguru import logger  # 导入日志模块

# 队列名称到队列ID的映射字典
QUEUE_NAME_TO_ID = {
    "llmeval_volc": "q-20241107085952-dnfrk",
    "mllm1": "q-20241107090119-5rpvq",
    # 可以添加更多队列名称映射
}

# 配置loguru日志记录器（默认只输出到控制台，不生成本地文件日志）
logger.remove()  # 移除默认处理器
logger.add(
    lambda msg: print(msg, flush=True),  # 只打印到控制台
    colorize=True,  # 颜色化输出
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # 控制台日志格式
)

def get_current_conda_env() -> str:
    """获取当前conda环境的路径。"""
    try:
        # 从环境中获取CONDA_PREFIX
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            logger.debug(f"从CONDA_PREFIX找到的conda环境: {conda_prefix}")
            return conda_prefix
        
        # 如果未设置CONDA_PREFIX，尝试从conda命令获取
        logger.debug("未找到CONDA_PREFIX，尝试使用conda info命令")
        result = subprocess.run(
            'conda info --envs | grep "*" | awk \'{print $NF}\'',  # 获取当前激活的conda环境
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            env_path = result.stdout.strip()
            logger.debug(f"从命令找到的conda环境: {env_path}")
            return env_path
        
    except Exception as e:
        logger.warning(f"检测conda环境失败: {e}")
    
    # 如果检测失败，返回默认环境
    default_env = '/fs-computility/llm/shared/llmeval/share_envs/oc-v034-ld-v061'
    logger.warning(f"使用默认的conda环境: {default_env}")
    return default_env

def load_storage_configs(yaml_path: str) -> List[Dict]:
    """从指定的YAML文件加载存储配置。"""
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if 'Storages' in yaml_data and isinstance(yaml_data['Storages'], list):
            logger.info(f"从 {yaml_path} 加载了 {len(yaml_data['Storages'])} 个存储配置")
            return yaml_data['Storages']
        else:
            logger.warning(f"在 {yaml_path} 中没有找到有效的存储配置")
            return []
    except Exception as e:
        logger.error(f"加载存储配置时出错: {str(e)}")
        return []

@dataclass
class RayConfig:
    """Ray分布式训练配置。"""
    task_name: str = 'ray-distributed-training'  # 任务名称
    queue_name: str = 'q-20241107085952-dnfrk'  # 队列ID
    framework: str = 'Custom'  # 框架类型（改为Custom）
    flavor: str = 'ml.hpcpni2l.28xlarge'  # 机器类型（默认8卡）
    n_nodes: int = 2  # 节点数
    n_gpus_per_node: int = 8  # 每个节点的GPU数量
    image: str = "fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest"  # Docker镜像
    active_deadline_seconds: int = 864000  # 活动期限（秒）
    extra_envs: list = field(default_factory=lambda: [  # 额外的环境变量
        # 'TORCH_NCCL_AVOID_RECORD_STREAMS=1',
        # 'VLLM_ATTENTION_BACKEND=XFORMERS',
        # 'VLLM_USE_V1=1',
        # 上面应该都是DAPO的那个参考代码中被o3复制过来的 
        # orz的vllm版本中这个不能用V1
        # 添加额外的环境变量
        'COMPASS_DATA_CACHE=/fs-computility/llm/shared/llmeval/datasets/compass_data_cache',
        'TORCH_HOME=/fs-computility/llm/shared/llmeval/torch',
        'TIKTOKEN_CACHE_DIR=/fs-computility/llm/shared/llmeval/share_tiktoken',
        # 添加离线模式环境变量
        'HF_DATASETS_OFFLINE=1',
        'TRANSFORMERS_OFFLINE=1',
        'HF_EVALUATE_OFFLINE=1',
        'HF_HUB_OFFLINE=1',
        'HF_ENDPOINT=https://hf-mirror.com'
    ])
    # 存储配置相关
    vepfs_id: str = ""  # VePFS ID (兼容旧版本，使用单个VePFS挂载)
    mount_path: str = "/file_system"  # 挂载路径 (兼容旧版本，使用单个VePFS挂载)
    storage_type: str = "Vepfs"  # 存储类型 (兼容旧版本，使用单个VePFS挂载)
    use_yaml_storage: bool = True  # 是否使用YAML配置中的存储配置
    volc_cfg_file: str = '/fs-computility/mllm1/limo/workspace/opencompass/configs/internal/volc_infer_limo.yaml'  # Volcano配置文件路径
    # 认证与跟踪
    volc_key_id: str = ""  # Volc访问密钥ID
    volc_secret_key: str = ""  # Volc访问密钥
    volc_region: str = ""  # Volc区域
    enable_tracking: bool = False  # 启用实验跟踪
    project_name: str = ""  # 项目名称
    experiment_name: str = ""  # 实验名称
    description: str = "Ray分布式训练任务"  # 任务描述
    # 缓存路径
    huggingface_cache: str = '/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub'  # Hugging Face缓存路径
    torch_cache: str = '/fs-computility/llm/shared/llmeval/torch'  # Torch缓存路径

class RayDeployment:
    """处理将ML任务部署到Ray分布式基础设施。"""
    
    def __init__(self, config: Optional[Dict] = None):
        """使用配置初始化部署。"""
        self.config = RayConfig(**config) if config else RayConfig()  # 初始化配置
        self.pwd = os.getcwd()  # 获取当前工作目录
        logger.info("使用配置初始化RayDeployment:")
        logger.info(f"工作目录: {self.pwd}")
        for key, value in asdict(self.config).items():
            if key != "volc_secret_key":  # 不打印密钥
                logger.info(f"{key}: {value}")
        
        # 加载存储配置
        self.storage_configs = []
        if self.config.use_yaml_storage and os.path.exists(self.config.volc_cfg_file):
            self.storage_configs = load_storage_configs(self.config.volc_cfg_file)
            logger.info(f"从YAML配置文件加载了{len(self.storage_configs)}个存储配置")
        
        # 保存当前的日志级别字符串
        self._log_level = "INFO"
        for handler in logger._core.handlers.values():
            if hasattr(handler, "levelno"):
                level_no = handler.levelno
                if level_no <= 10:  # DEBUG或更低
                    self._log_level = "DEBUG"
                    break
    
    def build_environment_variables(self) -> List[Dict]:
        """构建环境变量列表。"""
        envs = []
        
        # 添加动态环境变量
        dynamic_envs = [
            {"Name": "PYTHONPATH", "Value": f"{self.pwd}:$PYTHONPATH", "IsPrivate": False},
            {"Name": "HF_HUB_CACHE", "Value": self.config.huggingface_cache, "IsPrivate": False},
            {"Name": "HUGGINGFACE_HUB_CACHE", "Value": self.config.huggingface_cache, "IsPrivate": False},
            {"Name": "TORCH_HOME", "Value": self.config.torch_cache, "IsPrivate": False},
        ]
        
        envs.extend(dynamic_envs)
        
        # 添加基本环境变量
        for env in self.config.extra_envs:
            if "=" in env:
                name, value = env.split("=", 1)
                envs.append({"Name": name, "Value": value, "IsPrivate": False})
            else:
                envs.append({"Name": env, "Value": "1", "IsPrivate": False})
        
        # 如果启用了实验跟踪，添加跟踪相关环境变量
        if self.config.enable_tracking and self.config.volc_key_id and self.config.volc_secret_key:
            envs.extend([
                {"Name": "VOLC_ACCESS_KEY_ID", "Value": self.config.volc_key_id, "IsPrivate": False},
                {"Name": "VOLC_SECRET_ACCESS_KEY", "Value": self.config.volc_secret_key, "IsPrivate": True},
                {"Name": "MLP_TRACKING_REGION", "Value": self.config.volc_region or "cn-beijing", "IsPrivate": False},
            ])
            
            if self.config.project_name:
                envs.append({"Name": "MLP_TRACKING_PROJECT_NAME", "Value": self.config.project_name, "IsPrivate": False})
        
        return envs
    
    def build_storage_config(self) -> List[Dict]:
        """构建存储配置。"""
        logger.debug(f"构建存储配置: use_yaml_storage={self.config.use_yaml_storage}, yaml_exists={os.path.exists(self.config.volc_cfg_file)}, storage_configs_count={len(self.storage_configs)}")
        
        # 优先使用从YAML文件加载的存储配置
        if self.storage_configs and self.config.use_yaml_storage:
            logger.debug(f"使用从YAML加载的{len(self.storage_configs)}个存储配置")
            return self.storage_configs
        
        # 如果没有从YAML加载存储配置，但提供了vepfs_id，使用单个挂载点
        if self.config.vepfs_id:
            logger.debug(f"使用指定的vepfs_id: {self.config.vepfs_id}")
            return [
                {
                    "MountPath": self.config.mount_path,
                    "Type": self.config.storage_type,
                    "VepfsId": self.config.vepfs_id,
                }
            ]
        
        # 没有配置存储
        logger.debug("没有找到任何存储配置")
        return []
    
    def deploy_ray(self, task_cmd: str) -> subprocess.CompletedProcess:
        """将任务部署到Ray分布式训练基础设施。"""
        logger.info(f"开始部署Ray分布式训练，使用{self.config.n_nodes}个节点，每个节点{self.config.n_gpus_per_node}个GPU")
        logger.info(f"任务命令: {task_cmd}")
        
        try:
            # 准备Ray的配置
            ray_cfg = {
                "TaskName": self.config.task_name,
                "Description": self.config.description,
                "Entrypoint": task_cmd,
                "Envs": self.build_environment_variables(),
                "ResourceQueueID": self.config.queue_name,  # 确保这是队列ID而不仅仅是名称
                "Framework": self.config.framework,
                "TaskRoleSpecs": [
                    {
                        "RoleName": "worker",
                        "RoleReplicas": self.config.n_nodes,  # 所有节点都使用worker角色
                        "Flavor": self.config.flavor,
                    }
                ],
                "ActiveDeadlineSeconds": self.config.active_deadline_seconds,
                "EnableTensorBoard": False,
                "ImageUrl": self.config.image,
                "RetryOptions": {
                    "EnableRetry": False,
                    "MaxRetryTimes": 5,
                    "IntervalSeconds": 120,
                    "PolicySets": [],
                },
            }
            
            # 添加存储配置
            storage_config = self.build_storage_config()
            if storage_config:
                ray_cfg["Storages"] = storage_config
                logger.info(f"添加了{len(storage_config)}个存储配置")
                for i, storage in enumerate(storage_config):
                    logger.info(f"存储配置 {i+1}: {storage['Type']} -> {storage['MountPath']}")
            else:
                # 如果没有存储配置，尝试从YAML文件加载默认存储配置
                logger.warning("未找到有效的存储配置，尝试从YAML文件加载默认配置")
                default_storage = load_storage_configs(self.config.volc_cfg_file)
                if default_storage:
                    ray_cfg["Storages"] = default_storage
                    logger.info(f"从默认YAML加载了{len(default_storage)}个存储配置")
                    for i, storage in enumerate(default_storage):
                        logger.info(f"默认存储配置 {i+1}: {storage['Type']} -> {storage['MountPath']}")
                else:
                    logger.error("无法找到任何可用的存储配置，任务可能无法访问文件系统")
            
            os.makedirs(f'{self.pwd}/tmp', exist_ok=True)  # 创建临时目录
            tmp_cfg_file = f'{self.pwd}/tmp/{uuid.uuid4()}_ray_cfg.yaml'  # 生成临时配置文件名
            logger.debug(f"创建临时配置文件: {tmp_cfg_file}")
            
            with open(tmp_cfg_file, 'w') as fp:
                yaml.dump(ray_cfg, fp, sort_keys=False)  # 写入YAML配置
            
            # 打印生成的配置用于调试
            if self._log_level == 'DEBUG':  # 使用保存的日志级别
                with open(tmp_cfg_file, 'r') as fp:
                    logger.debug(f"生成的YAML配置:\n{fp.read()}")
            
            submit_cmd = f'volc ml_task submit --conf "{tmp_cfg_file}"'
            
            logger.info("提交Ray分布式训练任务")
            logger.debug(f"提交命令: {submit_cmd}")
            
            result = subprocess.run(
                submit_cmd,
                shell=True,
                text=True,
                capture_output=True
            )
            
            # 检查返回码
            if result.returncode != 0:
                logger.error(f"Ray分布式训练任务提交失败，返回码: {result.returncode}")
                if result.stderr:
                    logger.error(f"错误信息: {result.stderr}")
                if result.stdout:
                    logger.error(f"输出信息: {result.stdout}")
                raise subprocess.CalledProcessError(result.returncode, submit_cmd, 
                                                   result.stdout, result.stderr)
            
            logger.info("Ray分布式训练任务提交成功")
            if result.stdout:
                logger.info(f"输出: {result.stdout}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ray分布式训练部署失败: {str(e)}")
            # 如果是 CalledProcessError，打印更多信息
            if isinstance(e, subprocess.CalledProcessError) and hasattr(e, 'stderr') and e.stderr:
                logger.error(f"命令错误详情: {e.stderr}")
            raise
        finally:
            # 完成后可以选择清理临时文件
            pass

def extract_queue_id_from_yaml(yaml_path: str) -> str:
    """从YAML文件中提取队列ID。"""
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if 'ResourceQueueID' in yaml_data and isinstance(yaml_data['ResourceQueueID'], str):
            logger.info(f"从YAML文件中提取到队列ID: {yaml_data['ResourceQueueID']}")
            return yaml_data['ResourceQueueID']
        else:
            logger.warning(f"在YAML文件中未找到有效的队列ID")
            return ""
    except Exception as e:
        logger.error(f"尝试从YAML提取队列ID时出错: {str(e)}")
        return ""

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='将ML任务部署到Ray分布式基础设施')
    
    # 必需参数
    parser.add_argument('--task-cmd', required=True, help='要执行的主要任务命令')
    
    # Ray 相关参数
    parser.add_argument('--n-nodes', type=int, default=2, help='Ray集群的节点数（默认: 2）')
    parser.add_argument('--n-gpus-per-node', type=int, default=8, help='每个节点的GPU数量（默认: 8）')
    
    # 共享参数
    parser.add_argument('--task-name', help='覆盖默认任务名称')
    parser.add_argument('--queue-name', help='队列名称或ID。可以使用简单名称如 "llmeval_volc" 或 "mllm1"，将自动映射到对应的队列ID')
    parser.add_argument('--image', help="覆盖默认镜像")
    parser.add_argument('--extra-envs', nargs='+', help='额外的环境变量，格式为KEY=VALUE')
    
    # 存储相关
    parser.add_argument('--vepfs-id', help='VePFS ID（可选，优先使用YAML配置）')
    parser.add_argument('--mount-path', default='/file_system', help='挂载路径（默认: /file_system）')
    parser.add_argument('--storage-type', default='Vepfs', help='存储类型（默认: Vepfs）')
    parser.add_argument('--volc-cfg-file', 
                       default='/fs-computility/mllm1/limo/workspace/opencompass/configs/internal/volc_infer_limo.yaml',
                       help='YAML配置文件路径，用于读取存储配置')
    parser.add_argument('--no-yaml-storage', dest='use_yaml_storage', action='store_false', 
                       help='不使用YAML文件中的存储配置')
    
    # 实验跟踪
    parser.add_argument('--enable-tracking', action='store_true', help='启用实验跟踪')
    parser.add_argument('--project-name', help='项目名称（跟踪）')
    parser.add_argument('--experiment-name', help='实验名称（跟踪）')
    
    # 兼容性参数 - 只为了兼容旧命令，但实际不使用
    parser.add_argument('--deploy-type', choices=['ray'], default='ray',
                      help='部署类型: 只支持ray (默认: ray)')
    
    # 其他
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='设置日志级别（默认: INFO）')
    parser.add_argument('--yes', action='store_true', help='自动确认部署，不提示')
    
    return parser.parse_args()

def get_queue_id(queue_name: str) -> str:
    """
    将队列名称转换为队列ID。
    如果输入的已经是队列ID或者名称不在映射字典中，则直接返回输入值。
    """
    # 检查是否是已经以'q-'开头的ID
    if queue_name.startswith('q-'):
        return queue_name
    
    # 尝试从映射字典中获取ID
    queue_id = QUEUE_NAME_TO_ID.get(queue_name)
    if queue_id:
        logger.info(f"将队列名称 '{queue_name}' 映射到队列ID '{queue_id}'")
        return queue_id
    
    # 如果映射不存在，返回原始输入
    logger.warning(f"未找到队列名称 '{queue_name}' 的映射，将直接使用作为队列ID")
    return queue_name

def main():
    """主执行函数。"""
    args = parse_args()  # 解析命令行参数
    
    # 设置日志级别
    logger.remove()  # 移除现有处理器
    logger.add(
        lambda msg: print(msg, flush=True),  # 只打印到控制台
        colorize=True,
        level=args.log_level,  # 控制台日志级别
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"  # 控制台日志格式
    )
    
    logger.info("开始Ray部署脚本")
    
    # 获取当前conda环境
    current_env = get_current_conda_env()
    logger.info(f"当前conda环境: {current_env}")
    
    # Ray分布式部署
    # 获取环境变量中的认证信息
    volc_key_id = os.environ.get('VOLC_ACCESS_KEY_ID', '')
    volc_secret_key = os.environ.get('VOLC_SECRET_ACCESS_KEY', '')
    volc_region = os.environ.get('VOLC_REGION', '')
    
    # 关键修改：获取正确的队列ID
    volc_cfg_file = args.volc_cfg_file or '/fs-computility/mllm1/limo/workspace/opencompass/configs/internal/volc_infer_limo.yaml'
    queue_id = ""
    
    # 首先尝试从YAML文件中提取队列ID
    if os.path.exists(volc_cfg_file):
        queue_id = extract_queue_id_from_yaml(volc_cfg_file)
        if queue_id:
            logger.info(f"从YAML配置文件中提取到队列ID: {queue_id}")
        else:
            logger.warning("无法从YAML配置文件中提取队列ID")
    
    # 如果命令行指定了队列名称，使用命令行参数（可能需要转换）
    if args.queue_name:
        queue_id = get_queue_id(args.queue_name)
        logger.info(f"使用命令行指定的队列: {args.queue_name} -> {queue_id}")
    
    # 如果仍然没有队列ID，使用默认值
    if not queue_id:
        queue_id = "q-20241107085952-dnfrk"  # 使用YAML中的默认ID
        logger.warning(f"没有找到有效的队列ID，使用默认值: {queue_id}")
    
    # 准备Ray配置
    ray_config = {
        'task_name': args.task_name or f'ray-dist-{uuid.uuid4().hex[:8]}',
        'queue_name': queue_id,  # 使用提取出的队列ID
        'n_nodes': args.n_nodes,
        'n_gpus_per_node': args.n_gpus_per_node,
        'image': args.image or "fs-computility-cn-beijing.cr.volces.com/devinstance-archive/orz:latest",  # 更新默认镜像
        'volc_key_id': volc_key_id,
        'volc_secret_key': volc_secret_key,
        'volc_region': volc_region,
        'use_yaml_storage': args.use_yaml_storage,
        'volc_cfg_file': volc_cfg_file,
    }
    
    # 如果指定了VePFS ID，添加到配置中
    if args.vepfs_id:
        ray_config['vepfs_id'] = args.vepfs_id
        ray_config['mount_path'] = args.mount_path
        ray_config['storage_type'] = args.storage_type
    
    if args.extra_envs:
        ray_config['extra_envs'] = args.extra_envs
    
    if args.enable_tracking:
        ray_config['enable_tracking'] = True
        if args.project_name:
            ray_config['project_name'] = args.project_name
        if args.experiment_name:
            ray_config['experiment_name'] = args.experiment_name
    
    # 初始化Ray部署
    ray_deployer = RayDeployment(ray_config)
    
    # 打印部署配置
    logger.info("\nRay分布式训练配置摘要:")
    logger.info(f"任务命令: {args.task_cmd}")
    logger.info(f"节点数量: {args.n_nodes}")
    logger.info(f"每个节点的GPU数量: {args.n_gpus_per_node}")
    logger.info(f"总GPU数量: {args.n_nodes * args.n_gpus_per_node}")
    logger.info(f"任务名称: {ray_deployer.config.task_name}")
    logger.info(f"队列ID: {ray_deployer.config.queue_name}")
    logger.info(f"镜像名称: {ray_deployer.config.image}")
    logger.info(f"从YAML加载存储配置: {'是' if ray_deployer.config.use_yaml_storage else '否'}")
    logger.info(f"YAML配置文件: {ray_deployer.config.volc_cfg_file}")
    
    # 确认部署
    if not args.yes:
        confirm = input("\n继续部署Ray分布式训练任务吗？ [y/N]: ")
        if confirm.lower() != 'y':
            logger.warning("用户取消了部署")
            return
    else:
        logger.info("自动确认部署（--yes标志）")
    
    # 执行Ray部署
    try:
        result = ray_deployer.deploy_ray(args.task_cmd)
        
        # 打印部署结果
        if result.returncode == 0:
            logger.success("Ray分布式训练任务部署成功完成")
        else:
            logger.error("Ray分布式训练任务部署失败")
            
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        if result.stderr:
            logger.warning(f"错误: {result.stderr}")
            
    except Exception as e:
        logger.exception("Ray分布式训练任务部署失败，出现异常")
        raise

if __name__ == "__main__":
    main()  # 执行主函数