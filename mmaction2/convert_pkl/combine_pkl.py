import os
import os.path as osp
import pickle

result = []
path = '/home/jovyan/2024-srtp/mmaction2/k400-processed'  # 你的多个pickle文件夹路径

# 使用os.walk遍历所有子目录
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.pkl'):
            file_path = osp.join(root, file)
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                result.append(content)

# 将合并后的内容保存到新的pickle文件
with open('train.pkl', 'wb') as out:
    pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)

print("All .pkl files have been processed and merged into train.pkl")