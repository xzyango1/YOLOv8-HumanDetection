# data_preparation/remap_labels.py
import os
from pathlib import Path

# --- 配置区 ---
# ⚠️ 在运行前，请务必根据你解压后的文件夹名，修改下面这个列表！
# 这是你在 'datasets/' 目录下为三个原始数据集创建的文件夹的名字。
SOURCE_DATASET_DIRS = [
    'DS1_Safety_Helmet',       # 你的第一个安全帽数据集文件夹名
    'DS2_Safety_Helmet_New',   # 你的第二个安全帽数据集文件夹名
    'DS3_Human'                # 你的人体数据集文件夹名
]

# 我们的“主类别”标准。所有数据集都将被统一到这个标准。
# 0 = head, 1 = helmet, 2 = person
MASTER_CLASSES = {
    'head': 0,
    'helmet': 1,
    'person': 2,
}

# 为每个数据集定义它们的“原始语言”到“主类别”的翻译规则。
# key是原始类别ID, value是我们要转换成的目标类别ID。
REMAPPING_RULES = {
    'DS1_Safety_Helmet': {0: MASTER_CLASSES['head'], 1: MASTER_CLASSES['helmet'], 2: MASTER_CLASSES['person']},
    'DS2_Safety_Helmet_New': {0: MASTER_CLASSES['helmet'], 1: MASTER_CLASSES['head']},
    'DS3_Human': {0: MASTER_CLASSES['person']},
}

# --- 核心执行区 ---

def remap_class_ids_in_dir(labels_dir, id_mapping):
    """遍历指定目录下的所有.txt标签文件，并根据映射关系重写类别ID。"""
    if not labels_dir.exists():
        print(f"  - 🟡 警告: 目录不存在，已跳过: {labels_dir}")
        return 0

    remap_count = 0
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        file_was_remapped = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            try:
                old_id = int(parts[0])
                if old_id in id_mapping:
                    parts[0] = str(id_mapping[old_id])
                    new_lines.append(" ".join(parts) + "\n")
                    remap_count += 1
                    file_was_remapped = True
                else:
                    new_lines.append(line)
            except ValueError:
                print(f"  - 🔴 错误: 在文件 {label_file} 中发现非数字类别ID，已跳过此行。")
                new_lines.append(line)

        if file_was_remapped:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
    return remap_count

def main():
    """主函数，执行所有重映射任务。"""
    project_root = Path(__file__).resolve().parent.parent
    datasets_root = project_root / 'datasets'

    print("="*60)
    print("🚀 欢迎使用“标签重映射”脚本！")
    print("本脚本将自动统一所有源数据集的类别ID，为最终合并做准备。")
    print(f"👉 当前配置要处理的文件夹为: {SOURCE_DATASET_DIRS}")
    print("="*60)
    
    confirm = input("❓ 请确认以上文件夹名是否已在脚本顶部修改正确? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 操作已取消。请先修改脚本顶部的 `SOURCE_DATASET_DIRS` 列表。")
        return

    total_remapped = 0
    for dir_name in SOURCE_DATASET_DIRS:
        print(f"\n--- 正在处理数据集: {dir_name} ---")
        dataset_path = datasets_root / dir_name
        mapping_rule = REMAPPING_RULES.get(dir_name)

        if not dataset_path.exists():
            print(f"🔴 错误: 找不到数据集目录 -> {dataset_path}")
            continue
        if not mapping_rule:
            print(f"🔴 错误: 在 REMAPPING_RULES 中找不到 '{dir_name}' 的映射规则。")
            continue

        for split in ['train', 'valid', 'test']:
            labels_path = dataset_path / split / 'labels'
            print(f"  - 正在检查 '{split}' 部分...")
            count = remap_class_ids_in_dir(labels_path, mapping_rule)
            if count > 0:
                print(f"    ✅ 在 '{split}' 部分成功重映射了 {count} 个标签。")
            total_remapped += count

    print("\n" + "="*60)
    if total_remapped > 0:
        print(f"🎉 所有数据集的标签重映射已成功完成！总计修改了 {total_remapped} 个标签。")
    else:
        print("✅ 检查完成，所有标签均已符合标准，无需修改。")
    print("现在你可以运行 `merge_and_verify.py` 脚本了。")
    print("="*60)

if __name__ == '__main__':
    main()