#!/usr/bin/env python3
import os
import sys

def create_soft_links(obj_list_file):
    """
    为txt文件中的每个obj_name创建软链接
    从 DGN_5k -> select_100
    """
    
    # 读取obj_name列表
    with open(obj_list_file, 'r') as f:
        obj_names = [line.strip() for line in f if line.strip()]
    
    print(f"从文件中读取了 {len(obj_names)} 个物体名称")
    
    # 基础路径
    src_base = "assets/object/objaverse"
    dst_base = "assets/object/select_100"
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for obj_name in obj_names:
        # 定义源路径和目标路径
        src_processed = os.path.join(src_base, "processed_data", obj_name)
        src_scene = os.path.join(src_base, "scene_cfg", obj_name)
        
        dst_processed = os.path.join(dst_base, "processed_data", obj_name)
        dst_scene = os.path.join(dst_base, "scene_cfg", obj_name)
        
        # 检查源目录是否存在
        if not os.path.exists(src_processed):
            print(f"⚠️  跳过 {obj_name}: 源目录不存在 {src_processed}")
            skip_count += 1
            continue
            
        if not os.path.exists(src_scene):
            print(f"⚠️  跳过 {obj_name}: 源目录不存在 {src_scene}")
            skip_count += 1
            continue
        
        try:
            # 创建目标目录的父目录（如果不存在）
            os.makedirs(os.path.dirname(dst_processed), exist_ok=True)
            os.makedirs(os.path.dirname(dst_scene), exist_ok=True)
            
            # 创建软链接（如果已存在则先删除）
            if os.path.lexists(dst_processed):
                os.remove(dst_processed)
                print(f"🔄 已移除现有的软链接: {dst_processed}")
            
            if os.path.lexists(dst_scene):
                os.remove(dst_scene)
                print(f"🔄 已移除现有的软链接: {dst_scene}")
            
            # 创建相对路径的软链接（更易于移植）
            # 计算从目标到源的相对路径
            rel_processed = os.path.relpath(src_processed, os.path.dirname(dst_processed))
            rel_scene = os.path.relpath(src_scene, os.path.dirname(dst_scene))
            
            os.symlink(rel_processed, dst_processed)
            os.symlink(rel_scene, dst_scene)
            
            print(f"✅ {obj_name}: 软链接创建成功")
            success_count += 1
            
        except OSError as e:
            print(f"❌ {obj_name}: 创建失败 - {e}")
            error_count += 1
    
    print(f"\n完成！成功: {success_count}, 跳过: {skip_count}, 错误: {error_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python create_links.py <obj_list.txt>")
        print("示例: python create_links.py selected_objects.txt")
        sys.exit(1)
    
    txt_file = sys.argv[1]
    if not os.path.exists(txt_file):
        print(f"错误: 文件 '{txt_file}' 不存在")
        sys.exit(1)
    
    create_soft_links(txt_file)