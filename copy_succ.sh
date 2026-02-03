#!/bin/bash

# 设置基础路径
BASE_DST="output/Dexonomy_GRASP_small_shadow"
BASE_SRC="output/Dexonomy_GRASP_shadow"

# 遍历 graspdata 下的所有子文件夹
find "${BASE_DST}/graspdata" -mindepth 2 -maxdepth 2 -type d | while read -r graspdata_dir; do
    # 提取相对路径，例如: 1_Large_Diameter/0afbd2427ca84114b659e1d8cfcb647a_floating
    rel_path="${graspdata_dir#${BASE_DST}/graspdata/}"
    
    # 构造源和目标 succgrasp 路径
    src_dir="${BASE_SRC}/succgrasp/${rel_path}"
    dst_dir="${BASE_DST}/succgrasp/${rel_path}"
    
    # 检查源目录是否存在且非空
    if [ -d "$src_dir" ] && [ -n "$(ls -A "$src_dir" 2>/dev/null)" ]; then
        echo "发现非空源目录: $src_dir"
        echo "复制到: $dst_dir"
        
        # 创建目标目录（如果不存在）
        mkdir -p "$dst_dir"
        
        # 复制所有文件
        cp -r "$src_dir"/* "$dst_dir"/
        
        echo "复制完成"
        echo "-------------------"
    fi
done