# data_preparation/merge_and_verify.py
import os
import shutil
from pathlib import Path
from collections import defaultdict

# --- 配置区 ---
# ⚠️ 在运行前，请确保这里的文件夹名列表与 `remap_labels.py` 中的完全一致！
SOURCE_DATASET_DIRS = [
    'DS1_Safety_Helmet',
    'DS2_Safety_Helmet_New',
    'DS3_Human'
]

# 你的终极目标数据集的文件夹名称
ULTIMATE_DATASET_DIR = 'ULTIMATE_DATASET'

# --- 核心执行区 ---

def copy_files(source_dir, dest_dir):
    """一个安全的复制函数，只在源文件存在时进行复制。"""
    if not source_dir.exists():
        return 0
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for src_file in source_dir.iterdir():
        if src_file.is_file():
            shutil.copy2(str(src_file), str(dest_dir))
            copied_count += 1
    return copied_count

def main():
    """主函数，执行合并与验证。"""
    project_root = Path(__file__).resolve().parent.parent
    datasets_root = project_root / 'datasets'
    ultimate_path = datasets_root / ULTIMATE_DATASET_DIR

    print("="*60)
    print("🚀 欢迎使用“数据集合并与验证”脚本！")
    print("本脚本将把所有源数据集的文件，合并到一个统一的终极数据集中。")
    print(f"👉 将要合并的源文件夹: {SOURCE_DATASET_DIRS}")
    print(f"👉 将要创建/填充的目标文件夹: {ULTIMATE_DATASET_DIR}")
    print("="*60)

    confirm = input("❓ 确认开始执行合并吗? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ 操作已取消。")
        return

    # --- 阶段一: 合并文件 ---
    print("\n--- 阶段一: 开始合并文件 ---")
    if ultimate_path.exists():
        print(f"🟡 警告: 目标文件夹 '{ULTIMATE_DATASET_DIR}' 已存在。脚本将向其中添加文件。")
    
    for split in ['train', 'valid', 'test']:
        print(f"\n--- 正在处理 '{split}' 部分 ---")
        for source_name in SOURCE_DATASET_DIRS:
            print(f"  -> 从 '{source_name}' 复制...")
            
            source_images = datasets_root / source_name / split / 'images'
            source_labels = datasets_root / source_name / split / 'labels'
            
            dest_images = ultimate_path / split / 'images'
            dest_labels = ultimate_path / split / 'labels'

            img_count = copy_files(source_images, dest_images)
            lbl_count = copy_files(source_labels, dest_labels)
            print(f"     ✅ 复制了 {img_count} 个图片, {lbl_count} 个标签。")
            
    print("\n--- ✅ 所有数据集合并完成！ ---")

    # --- 阶段二: 验证文件数量 ---
    print("\n\n--- 阶段二: 开始验证文件数量 ---")
    totals = defaultdict(lambda: defaultdict(int))
    
    for split in ['train', 'valid', 'test']:
        for source_name in SOURCE_DATASET_DIRS:
            source_images = datasets_root / source_name / split / 'images'
            source_labels = datasets_root / source_name / split / 'labels'
            if source_images.exists(): totals[split]['source_images'] += len(list(source_images.glob('*.*')))
            if source_labels.exists(): totals[split]['source_labels'] += len(list(source_labels.glob('*.txt')))

        dest_images = ultimate_path / split / 'images'
        dest_labels = ultimate_path / split / 'labels'
        if dest_images.exists(): totals[split]['dest_images'] = len(list(dest_images.glob('*.*')))
        if dest_labels.exists(): totals[split]['dest_labels'] = len(list(dest_labels.glob('*.txt')))
            
    all_ok = True
    for split in ['train', 'valid', 'test']:
        print(f"\n--- 验证 '{split}' 部分 ---")
        src_img, dst_img = totals[split]['source_images'], totals[split]['dest_images']
        img_status = "✅ 匹配成功" if src_img == dst_img else f"❌ 失败 ({dst_img}/{src_img})"
        if src_img != dst_img: all_ok = False
        print(f"  图片文件: 源总数={src_img}, 目标总数={dst_img}  -> {img_status}")

        src_lbl, dst_lbl = totals[split]['source_labels'], totals[split]['dest_labels']
        lbl_status = "✅ 匹配成功" if src_lbl == dst_lbl else f"❌ 失败 ({dst_lbl}/{src_lbl})"
        if src_lbl != dst_lbl: all_ok = False
        print(f"  标签文件: 源总数={src_lbl}, 目标总数={dst_lbl}  -> {lbl_status}")
        
    print("\n" + "="*60)
    if all_ok:
        print("🎉 恭喜！所有文件均已正确合并！您的终极数据集已准备就绪！")
        print("现在你可以去修改并运行 `train.py` 来开始训练了。")
    else:
        print("🔥 注意：发现文件数量不匹配。请检查上面的报告。")
        print("   可能的原因是目标文件夹中存在上次运行的旧文件。")
        print("   建议删除整个 `ULTIMATE_DATASET` 文件夹后重试。")
    print("="*60)

if __name__ == "__main__":
    main()