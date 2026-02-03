import os
import random

def select_random_folders(base_path, num_folders):
    """
    从指定路径中随机选择指定数量的文件夹
    
    Args:
        base_path: 基础目录路径
        num_folders: 需要选择的文件夹数量
    
    Returns:
        list: 选中的文件夹名称列表
    """
    try:
        # 获取所有子文件夹
        all_folders = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d))]
        
        if len(all_folders) < num_folders:
            print(f"警告: {base_path} 中只有 {len(all_folders)} 个文件夹，"
                  f"少于要求的 {num_folders} 个")
            return all_folders
        
        # 随机选择指定数量的文件夹
        selected_folders = random.sample(all_folders, num_folders)
        return selected_folders
        
    except FileNotFoundError:
        print(f"错误: 路径 '{base_path}' 不存在")
        return []
    except Exception as e:
        print(f"错误: 读取路径 '{base_path}' 时发生意外错误: {e}")
        return []

def main():
    # 配置路径
    dgn_path = "assets/object/DGN_5k/processed_data"
    objaverse_path = "assets/object/objaverse_5k/processed_data"
    
    # 配置输出文件
    output_file = "selected_folders.txt"
    
    # 从每个目录选择50个文件夹
    print("开始选择文件夹...")
    dgn_folders = select_random_folders(dgn_path, 50)
    objaverse_folders = select_random_folders(objaverse_path, 50)
    
    # 合并结果
    all_selected_folders = dgn_folders + objaverse_folders
    
    if len(all_selected_folders) == 0:
        print("错误: 没有选中的文件夹，请检查路径是否正确")
        return
    
    # 保存到文件
    try:
        with open(output_file, 'w') as f:
            for folder in all_selected_folders:
                f.write(folder + '\n')
        
        print(f"\n成功选择 {len(all_selected_folders)} 个文件夹")
        print(f"结果已保存到: {output_file}")
        
        # 显示前10个作为示例
        print(f"\n前10个文件夹示例:")
        for i, folder in enumerate(all_selected_folders[:10], 1):
            print(f"{i}. {folder}")
            
    except Exception as e:
        print(f"错误: 写入文件时发生错误: {e}")

if __name__ == "__main__":
    main()