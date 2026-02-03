#!/bin/bash

# 设置基础路径
SOURCE_BASE="/mnt/home/chenjiayi/data/dexonomy_output/Dexonomy_GRASP_shadow/succgrasp"
DEST_BASE="/mnt/home/ruanliangwang/Dexonomy-private/output/test_100_shadow/succgrasp"
SELECTED_FILE="selected_folders.txt"

# 检查文件是否存在
if [[ ! -f "$SELECTED_FILE" ]]; then
    echo "错误：找不到 $SELECTED_FILE"
    exit 1
fi

# 创建目标基础目录（如果不存在）
mkdir -p "$DEST_BASE"

# 统计处理结果
TOTAL_OBJS=0
TOTAL_LINKS_CREATED=0
SKIPPED_OBJS=0

echo "开始处理..."
echo "========================================"

# 逐行读取 selected_folders.txt
while IFS= read -r obj; do
    # 跳过空行
    [[ -z "$obj" ]] && continue
    
    TOTAL_OBJS=$((TOTAL_OBJS + 1))
    
    # 构建目标文件夹名
    target_folder="${obj}_floating"
    
    # 当前obj找到的匹配数
    found_count=0
    
    # 遍历 SOURCE_BASE 下的所有template文件夹
    for template_dir in "$SOURCE_BASE"/*; do
        # 检查是否是目录
        if [[ -d "$template_dir" ]]; then
            # 获取template文件夹名
            template_name=$(basename "$template_dir")
            
            # 源路径
            source_path="$template_dir/$target_folder"
            
            # 如果源路径存在
            if [[ -d "$source_path" ]]; then
                # 创建目标目录
                dest_template_dir="$DEST_BASE/$template_name"
                mkdir -p "$dest_template_dir"
                
                # 目标软链接路径
                dest_link="$dest_template_dir/$target_folder"
                
                # 如果已存在同名文件/链接，先删除
                if [[ -e "$dest_link" ]]; then
                    rm -rf "$dest_link"
                fi
                
                # 创建软链接（使用绝对路径）
                ln -s "$source_path" "$dest_link"
                
                found_count=$((found_count + 1))
                TOTAL_LINKS_CREATED=$((TOTAL_LINKS_CREATED + 1))
                
                echo "✓ [$found_count] $template_name/ → $target_folder"
            fi
        fi
    done
    
    # 输出当前obj的处理结果
    if [[ $found_count -eq 0 ]]; then
        echo "[$TOTAL_OBJS] $obj: 未找到任何匹配 ❌"
        SKIPPED_OBJS=$((SKIPPED_OBJS + 1))
    else
        echo "[$TOTAL_OBJS] $obj: 找到 $found_count 个匹配 ✅"
    fi
    echo ""
    
done < "$SELECTED_FILE"

echo "========================================"
echo "处理完成！"
echo "总计 obj 数量: $TOTAL_OBJS"
echo "成功创建软链接: $TOTAL_LINKS_CREATED"
echo "未找到匹配的 obj: $SKIPPED_OBJS"