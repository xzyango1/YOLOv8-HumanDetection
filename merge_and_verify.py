import os
import shutil
from collections import defaultdict

# --- 配置区 ---
# 请根据您的实际文件夹名称，仔细配置以下变量

# 1. 数据集的基础路径
BASE_PATH = "D:/VScode Project/YOLOv8-HumanDetection/datasets/"

# 2. 您所有源数据集的文件夹名称列表
SOURCE_DATASETS = [
    "Safety-Helmet-1",  # 您的第一个安全帽数据集
    "Safety-Helmet-2",  # 您下载的第二个安全帽数据集
    "Person-Detection-1",  # 您下载的人体数据集
]

# 3. 您的终极目标数据集的文件夹名称
ULTIMATE_DATASET = "ULTIMATE_DATASET"

# --- 核心功能代码 ---


def copy_files(source_dir, dest_dir):
    """一个安全的复制函数，只在源文件存在时进行复制"""
    if not os.path.exists(source_dir):
        print(f"  - 警告: 源文件夹不存在，跳过: {source_dir}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, dest_file)
            copied_count += 1
    return copied_count


def merge_datasets():
    """主合并函数"""
    print("--- 开始合并数据集 ---")
    dest_path_base = os.path.join(BASE_PATH, ULTIMATE_DATASET)

    for split in ["train", "valid", "test"]:
        print(f"\n--- 正在处理 '{split}' 部分 ---")

        for source_name in SOURCE_DATASETS:
            print(f"  -> 从 '{source_name}' 复制...")

            # 定义源和目标的 images 和 labels 路径
            source_images_path = os.path.join(BASE_PATH, source_name, split, "images")
            source_labels_path = os.path.join(BASE_PATH, source_name, split, "labels")

            dest_images_path = os.path.join(dest_path_base, split, "images")
            dest_labels_path = os.path.join(dest_path_base, split, "labels")

            # 执行复制
            img_count = copy_files(source_images_path, dest_images_path)
            lbl_count = copy_files(source_labels_path, dest_labels_path)
            print(f"     复制了 {img_count} 个图片, {lbl_count} 个标签。")

    print("\n--- 所有数据集合并完成！ ---")


def verify_merge():
    """主验证函数"""
    print("\n\n--- 开始验证合并结果 ---")

    totals = defaultdict(lambda: defaultdict(int))

    # 1. 统计所有源文件的总数
    for split in ["train", "valid", "test"]:
        for source_name in SOURCE_DATASETS:
            source_images_path = os.path.join(BASE_PATH, source_name, split, "images")
            source_labels_path = os.path.join(BASE_PATH, source_name, split, "labels")

            if os.path.exists(source_images_path):
                totals[split]["source_images"] += len(os.listdir(source_images_path))
            if os.path.exists(source_labels_path):
                totals[split]["source_labels"] += len(os.listdir(source_labels_path))

    # 2. 统计终极目标文件夹中的文件总数
    dest_path_base = os.path.join(BASE_PATH, ULTIMATE_DATASET)
    for split in ["train", "valid", "test"]:
        dest_images_path = os.path.join(dest_path_base, split, "images")
        dest_labels_path = os.path.join(dest_path_base, split, "labels")

        if os.path.exists(dest_images_path):
            totals[split]["dest_images"] = len(os.listdir(dest_images_path))
        if os.path.exists(dest_labels_path):
            totals[split]["dest_labels"] = len(os.listdir(dest_labels_path))

    # 3. 打印清晰的对比报告
    all_ok = True
    for split in ["train", "valid", "test"]:
        print(f"\n--- 验证 '{split}' 部分 ---")

        # 验证图片
        src_img = totals[split]["source_images"]
        dst_img = totals[split]["dest_images"]
        img_status = "✅ 匹配成功" if src_img == dst_img else "❌ 失败"
        if src_img != dst_img:
            all_ok = False
        print(f"  源图片总数: {src_img}")
        print(f"  目标图片总数: {dst_img}  -> {img_status}")

        # 验证标签
        src_lbl = totals[split]["source_labels"]
        dst_lbl = totals[split]["dest_labels"]
        lbl_status = "✅ 匹配成功" if src_lbl == dst_lbl else "❌ 失败"
        if src_lbl != dst_lbl:
            all_ok = False
        print(f"  源标签总数: {src_lbl}")
        print(f"  目标标签总数: {dst_lbl}  -> {lbl_status}")

    print("\n--- 验证完成 ---")
    if all_ok:
        print("🎉 恭喜！所有文件均已正确合并！您的数据集已准备就绪！")
    else:
        print("🔥 注意：发现文件数量不匹配。请检查上面的报告和您的文件夹。")


if __name__ == "__main__":
    # 第一步：执行合并
    merge_datasets()

    # 第二步：执行验证
    verify_merge()
