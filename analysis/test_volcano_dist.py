import os
import socket
import torch
import torch.distributed as dist

def print_env_vars():
    """打印所有MLP_前缀的环境变量"""
    print("=== 环境变量 ===")
    for k, v in sorted(os.environ.items()):
        if k.startswith("MLP_") or k in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            print(f"{k}: {v}")

def get_ip_addresses():
    """获取本机所有IP地址"""
    hostname = socket.gethostname()
    ip_list = socket.gethostbyname_ex(hostname)[2]
    return hostname, ip_list

def init_process_group():
    """初始化进程组并测试通信"""
    try:
        # 设置PyTorch分布式环境变量
        if "MLP_WORKER_0_HOST" in os.environ and not os.environ.get("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = os.environ["MLP_WORKER_0_HOST"]
            print(f"设置MASTER_ADDR={os.environ['MASTER_ADDR']}")
        
        if not os.environ.get("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "29500"
            
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        
        print(f"初始化进程组: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'Not set')}")
        print(f"MASTER_PORT={os.environ.get('MASTER_PORT', 'Not set')}")
        
        # 尝试初始化进程组
        dist.init_process_group("nccl")
        print(f"进程组初始化成功! rank={dist.get_rank()}, world_size={dist.get_world_size()}")
        
        # 简单的集体通信测试
        tensor = torch.ones(1, device=f"cuda:{local_rank}") * rank
        dist.all_reduce(tensor)
        print(f"All-reduce结果: {tensor.item()} (应该是{sum(range(world_size))})")
        
        return True
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return False

def main():
    """主函数"""
    hostname, ip_list = get_ip_addresses()
    print(f"主机名: {hostname}")
    print(f"IP地址: {ip_list}")
    
    print_env_vars()
    
    # 测试本机到MASTER_ADDR的连接
    if "MLP_WORKER_0_HOST" in os.environ:
        master_host = os.environ["MLP_WORKER_0_HOST"]
        master_port = 29500
        print(f"测试连接到 {master_host}:{master_port}...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((master_host, master_port))
            s.close()
            print(f"成功连接到 {master_host}:{master_port}")
        except Exception as e:
            print(f"无法连接到 {master_host}:{master_port}: {e}")
    
    # 初始化分布式进程组
    success = init_process_group()
    
    # 等待所有进程
    if success and dist.is_initialized():
        dist.barrier()
    
    print("测试完成!")

if __name__ == "__main__":
    main() 