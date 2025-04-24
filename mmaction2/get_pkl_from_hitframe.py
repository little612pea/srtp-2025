import json
import pickle
import numpy as np
import os
import argparse


def main(json_path, output_path):
    # === 读取 JSON 文件 ===
    with open(json_path, 'r') as f:
        data = json.load(f)

    joints = np.array(data['joints'])  # (221, 2, 17, 2)
    hit_frames = data['hit frames']
    start_frame = int(data['start frame'])

    frame_interval = 3  # joints 每 3 帧采一帧
    hit_indices = [int((frame - start_frame) / frame_interval) for frame in hit_frames]

    annotations = []

    for i, idx in enumerate(hit_indices):
        start = max(0, idx - 5)
        end = min(joints.shape[0], idx + 5)

        selected = joints[start:end]  # 最多 10 帧

        # 补齐不足 10 帧的情况
        if selected.shape[0] < 10:
            pad_len = 10 - selected.shape[0]
            pad = np.repeat(selected[-1:], pad_len, axis=0)
            selected = np.concatenate([selected, pad], axis=0)

        # 插值到 30 帧
        selected_interp = []
        for j in range(len(selected) - 1):
            start_frame = selected[j]
            end_frame = selected[j + 1]
            for t in np.linspace(0, 1, 3, endpoint=False):
                interp = start_frame * (1 - t) + end_frame * t
                selected_interp.append(interp)
        selected_interp.append(selected[-1])  # 加最后一帧，30 帧完成

        keypoints = np.stack(selected_interp)                  # (30, 2, 17, 2)
        keypoints = keypoints.transpose(1, 0, 2, 3)            # (2, 30, 17, 2)
        scores = np.ones((2, 30, 17), dtype=np.float32)

        # frame_dir 格式：A003_idx
        frame_dir_name = f"A003_{i:03d}"

        anno = {
            "frame_dir": frame_dir_name,
            "label": -1,  # 缺省
            "img_shape": (1080, 1920),
            "original_shape": (1080, 1920),
            "total_frames": keypoints.shape[1],
            "keypoint": keypoints.astype(np.float32),
            "keypoint_score": scores
        }

        annotations.append(anno)

    # === 构建 split 字典 ===
    split = {
        "train": [],
        "val": [anno["frame_dir"] for anno in annotations]
    }

    # === 最终结构 ===
    output_dict = {
        "split": split,
        "annotations": annotations
    }

    # === 保存为 pkl ===
    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)

    print(f"✅ 成功生成 hitting pkl，击球动作数: {len(annotations)}，输出文件：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rally JSON to PKL format for MMAction2.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, default="output_hitting.pkl", help="Path to save the output PKL file.")
    args = parser.parse_args()

    main(args.json_path, args.output_path)