# download dataset from huggingface
from huggingface_hub import snapshot_download
import os

# 设置镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载数据集（以 "csv" 格式数据集为例）
snapshot_download(
    repo_id="JiayiChenPKU/Dexonomy",  # 数据集名称
    repo_type="dataset",         # 类型为数据集
    cache_dir="/mnt/afs/ruanliangwang/cache",          # 下载到本地目录
    local_dir="/mnt/afs/ruanliangwang/Dexonomy-dataset",  # 直接保存到指定路径
    local_dir_use_symlinks=False  # 避免使用符号链接
)
