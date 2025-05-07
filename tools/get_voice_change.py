from moviepy import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt

def rms(audio_array):
    """计算音频片段的RMS值"""
    return np.sqrt(np.mean(np.square(audio_array), axis=1))

# 加载视频文件
video = VideoFileClip("/home/jovyan/2024-srtp/srtp-final/giting-with-voice.mp4")

# 提取音频
audio = video.audio.to_soundarray()

# 如果音频不是单声道，将其转换为单声道
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# 计算音频的RMS值
rms_values = rms(audio.reshape(-1, int(video.audio.fps))) # 假设fps为每秒帧数

# 时间轴
time = np.linspace(0, video.duration, len(rms_values))

# 绘制图形
plt.figure(figsize=(20, 6))
plt.plot(time, rms_values)
plt.title('Volume Change Over Time')
plt.xlabel('Time (s)')
plt.ylabel('RMS')
plt.grid(True)
plt.savefig('volume_change.png')