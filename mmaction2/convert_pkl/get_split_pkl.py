import pickle
#按照自己的方式切割训练集和验证集



import os
import pickle
import random
from collections import defaultdict
import numpy as np

# 设置文件夹路径
input_folder = '/home/jovyan/2024-srtp/mmaction2/k400-processed'  # 替换为你自己的路径
output_file = './annotations.pkl'  # 输出的文件路径

# 初始化存储xsub_train和xsub_val的字典
annotations = defaultdict(list)

# 遍历文件夹，获取所有子文件夹下的.pkl文件
def collect_pkl_files(root_folder):
    pkl_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pkl'):
                # 只添加文件名，不包含路径
                file_name = os.path.basename(file)
                print(file_name)
                file_name = file_name[:-4]
                file_name = file_name[:-4]
                print(file_name)
                pkl_files.append(file_name)
    return pkl_files

# 划分文件为训练集和验证集（8:2比例）
def split_files(pkl_files):
    random.shuffle(pkl_files)
    split_index = int(0.8 * len(pkl_files))
    train_files = pkl_files[:split_index]
    val_files = pkl_files[split_index:]
    return train_files, val_files



# 收集所有.pkl文件
pkl_files = collect_pkl_files(input_folder)

# 划分数据集
train_files, val_files = split_files(pkl_files)

# 添加到对应的字典部分
split = {
    'train': [os.path.basename(file) for file in train_files],
    'val': [os.path.basename(file) for file in val_files]
}


with open('train.pkl', 'rb') as f:
    annotations=pickle.loads(f.read())
with open('train_final.pkl', 'wb') as f:
        pickle.dump(dict(split=split, annotations=annotations), f)