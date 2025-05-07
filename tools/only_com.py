import os
import re

# 判断是否是英文字幕行（包含英文字母超过一定比例）
def is_english_line(line):
    # 匹配是否有英文字母
    return bool(re.search(r'[a-zA-Z]', line))

# 清理每一行，返回有效中文内容或 None
def clean_srt_line(line):
    line = line.strip()
    # 去除空行、序号、时间戳
    if not line:
        return None
    if re.match(r'^\d+$', line):
        return None
    if re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', line):
        return None
    # 过滤英文为主的行
    if is_english_line(line):
        return None
    return line

# 处理单个 srt 文件
def process_srt_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned = clean_srt_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(cleaned_lines))

# 批量处理文件夹中的 srt 文件
def process_all_srt_files(folder_path):
    output_folder = os.path.join(folder_path, 'output_txt')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith('.srt'):
            input_path = os.path.join(folder_path, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)
            process_srt_file(input_path, output_path)
            print(f"Processed: {filename} -> {output_filename}")

if __name__ == '__main__':
    folder_path = '/home/jovyan/2024-srtp/srtp-final/4K50FPS/srts'  # 替换为你的 .srt 文件所在文件夹路径
    process_all_srt_files(folder_path)