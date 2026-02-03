#!/bin/bash

SRC_BASE="output/Dexonomy_GRASP_shadow/succgrasp"
DEST_BASE="output/Dexonomy_GRASP_small_shadow/succgrasp"

for src_subdir in "$SRC_BASE"/*/; do
    # 获取子目录名称（如 "1_Large_Diameter"）
    subdir_name=$(basename "$src_subdir")
    
    # 构建目标子目录路径
    dest_subdir="$DEST_BASE/$subdir_name"
    
    # 创建目标子目录（如果不存在）
    mkdir -p "$dest_subdir"
    
    # 复制前100个文件/目录（按名称排序）
    find "$src_subdir" -maxdepth 1 -mindepth 1 | head -n 100 | while read item; do
        cp -r "$item" "$dest_subdir/"
    done
    
    echo "已复制 $subdir_name 下的前100个项目到 $dest_subdir"
done