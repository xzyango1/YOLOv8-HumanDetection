import os


def remap_class_ids(labels_dir, id_mapping):
    if not os.path.isdir(labels_dir):
        print(f"⚠️ 警告: 目录不存在，跳过处理 -> {labels_dir}")
        return
    print(f"开始处理目录: {labels_dir}...")
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()
            new_lines = []
            remapped = False
            for line in lines:
                parts = line.strip().split()
                if parts:
                    old_id = int(parts[0])
                    if old_id in id_mapping:
                        parts[0] = str(id_mapping[old_id])
                        new_lines.append(" ".join(parts) + "\n")
                        remapped = True
                    else:
                        new_lines.append(line)
            if remapped:
                with open(file_path, "w") as f:
                    f.writelines(new_lines)
    print("处理完成！")


if __name__ == "__main__":
    # --- 配置区 ---
    base_path = "D:/VScode Project/YOLOv8-HumanDetection/datasets/"

    # --- 任务1: 重映射 DS2 (新的安全帽数据集) ---
    print("--- 开始重映射 DS2 ---")
    ds2_path = os.path.join(base_path, "Safety-Helmet-2")
    ds2_mapping = {0: 1, 1: 0}  # Helmet -> helmet  # No_Helmet -> head
    remap_class_ids(os.path.join(ds2_path, "train/labels"), ds2_mapping)
    remap_class_ids(os.path.join(ds2_path, "valid/labels"), ds2_mapping)
    # 如果存在test集，也一并处理
    if os.path.exists(os.path.join(ds2_path, "test/labels")):
        remap_class_ids(os.path.join(ds2_path, "test/labels"), ds2_mapping)
    print("--- DS2 重映射完成 ---\n")

    # --- 任务2: 重映射 DS3 (人体数据集) ---
    print("--- 开始重映射 DS3 ---")
    ds3_path = os.path.join(base_path, "Person-Detection-1")
    ds3_mapping = {0: 2}  # human -> person
    remap_class_ids(os.path.join(ds3_path, "train/labels"), ds3_mapping)
    remap_class_ids(os.path.join(ds3_path, "valid/labels"), ds3_mapping)
    if os.path.exists(os.path.join(ds3_path, "test/labels")):
        remap_class_ids(os.path.join(ds3_path, "test/labels"), ds3_mapping)
    print("--- DS3 重映射完成 ---")
