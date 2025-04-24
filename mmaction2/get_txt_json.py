import pickle
import json
import numpy as np
import torch  # 添加对 torch 的支持（如果用不上也不影响）

def convert_to_native_types(obj):
    """
    将任意复杂数据结构（如 numpy / torch）转换为原生 Python 类型，便于 JSON 序列化。
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

# === 输入 PKL 路径 ===
pkl_path = '/home/jovyan/2024-srtp/srtp-final/mmaction2/temp.pkl'

# === 读取 PKL ===
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# === 转换为原生类型 ===
converted = convert_to_native_types(data)

# === 输出 JSON ===
json_path = 'output_temp.json'
with open(json_path, 'w') as f:
    json.dump(converted, f, indent=4)

print(f'已保存为 JSON：{json_path}')
