import json

# === 标签列表（按你给的顺序） ===
# label_names = [
#     "Short_Serve",
#     "Cross_Court_Flight",
#     "Lift",
#     "Tap_Smash",
#     "Block",
#     "Drop_Shot",
#     "Push_Shot",
#     "Transitional_Slice",
#     "Cut",
#     "Rush_Shot",
#     "Defensive_Clear",
#     "Defensive_Drive",
#     "Clear",
#     "Long_Serve",
#     "Smash",
#     "Flat_Shot",
#     "Rear_Court_Flat_Drive",
#     "Short_Flat_Shot"
# ]

label_names=[
"net shot",
"lift",
"smash",
"defensive drive",
"clear",
"flat shot"
]

# === 读取你的 JSON 文件 ===
with open("output_temp.json", "r") as f:
    data = json.load(f)

# === 添加可读的标签名 ===
for item in data:
    pred_idx = item["pred_label"][0]  # 取出类别索引
    item["pred_label_name"] = label_names[pred_idx]  # 添加对应名称

# === 保存为新的 JSON（可选） ===
with open("output_with_names.json", "w") as f:
    json.dump(data, f, indent=4)

# === 打印结果（示例） ===
for i, item in enumerate(data):
    print(f"样本 {i}: 预测类别为 {item['pred_label'][0]} → {item['pred_label_name']}")
