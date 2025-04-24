import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === 设置参数 ===
pkl_file = 'output_hitting.pkl'   # 你的 pkl 文件
output_dir = 'output_gifs'        # 输出文件夹
width, height = 1920, 1080        # 图像尺寸
conf_threshold = 0.7              # 显示关键点的置信度阈值

os.makedirs(output_dir, exist_ok=True)

# === 加载 pkl 文件 ===
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for anno in data['annotations']:
    frame_dir = anno['frame_dir']
    keypoints = anno['keypoint'].copy()         # shape: (2, 30, 17, 2)
    scores = anno['keypoint_score']             # shape: (2, 30, 17)
    

    # 可视化设置
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    scatters = [ax.plot([], [], 'bo', markersize=5)[0] for _ in range(2)]

    # 更新函数
    def update(frame_idx):
        for i, scatter in enumerate(scatters):
            print(f"Processing frame {frame_idx} for {frame_dir}...")
            print("i:", i)
            pts = keypoints[i, frame_idx]  # (17, 2)
            scatter.set_data(pts[:, 0], pts[:, 1])
        return scatters

    ani = FuncAnimation(fig, update, frames=28, interval=100, blit=True)

    # 保存 GIF
    output_path = os.path.join(output_dir, f'{frame_dir}.gif')
    ani.save(output_path, writer='Pillow', fps=30)
    plt.close(fig)  # 关闭当前窗口避免占用内存

    print(f"保存动画：{output_path}")

print("所有关键点动画已保存完毕。")
