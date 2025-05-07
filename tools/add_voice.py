import json
import subprocess
import os

# 定义路径和配置
json_file_path = "/home/jovyan/2024-srtp/srtp-final/hit_frame_detection/outputs/rallies/input_video.json"
input_dir = "/home/jovyan/2024-srtp/srtp-final/hit_frame_detection/outputs/videos/input_video"  # 替换为你的MP4文件所在目录
audio_file_path = "/home/jovyan/2024-srtp/srtp-final/audio-only.mp3"  # 替换为你的纯音频文件路径
output_dir = "/home/jovyan/2024-srtp/srtp-final/hit_frame_detection/outputs/videos/input_video/new"  # 替换为你希望保存输出文件的目录
frame_rate = 30  # 视频帧率


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载JSON文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 解析JSON数据，假设结构是 [[start, end, rally_id], ...]
audio_timestamps = data["rally"]

# 构建 rally_id -> 时间戳 映射
timestamp_map = {int(rally_id): (start, end) for start, end, rally_id in audio_timestamps}

# 遍历每个需要处理的视频文件
for filename in os.listdir(input_dir):
    if filename.endswith("h264.mp4"):
        print(f"Processing video: {filename}")
        
        # 提取 rally_id，例如 "rally_0_h264.mp4" → rally_id = 0
        try:
            rally_id = int(filename.split('_')[1])  # 假设文件名格式是 rally_X_h264.mp4
        except (IndexError, ValueError) as e:
            print(f"Could not extract rally_id from filename: {filename}. Error: {e}")
            continue

        # 查找对应的音频时间戳
        if rally_id not in timestamp_map:
            print(f"No matching timestamp found for rally_id={rally_id}")
            continue

        start_frame, end_frame = timestamp_map[rally_id]
        start_time = start_frame / frame_rate  # 转换为秒
        end_time = end_frame / frame_rate

        # 定义视频路径
        video_path = os.path.join(input_dir, filename)
        
        temp_audio_file = os.path.join(output_dir, f"temp_audio_{rally_id}.mp3")
        output_video_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mp4")

        # 使用ffmpeg从纯音频文件中提取音频片段
        cmd_extract_audio = [
            'ffmpeg', '-i', audio_file_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-q:a', '2',
            '-y',  # 覆盖已存在的文件
            temp_audio_file
        ]
        print(f"Extracting audio for rally_id={rally_id} from {start_time:.2f}s to {end_time:.2f}s")
        subprocess.run(cmd_extract_audio, check=True)

        # 合并音频片段到视频
        cmd_merge_audio_video = [
            'ffmpeg', '-i', video_path,
            '-i', temp_audio_file,
            '-c:v', 'copy',
            '-c:a', 'aac', '-strict', 'experimental',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            '-y',  # 覆盖已存在的文件
            output_video_path
        ]
        print(f"Merging audio into video: {filename}")
        subprocess.run(cmd_merge_audio_video, check=True)

        # 删除临时音频文件
        os.remove(temp_audio_file)
        print(f"Finished processing {filename}\n")

print("All videos processed.")