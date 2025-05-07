from tools.qwen_model import *
from tools.rag_with_qwen import *
import gradio as gr
import json
from openai import OpenAI  # 假设 OpenAI 模块可用
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import shutil
from pathlib import Path
import subprocess
from moviepy import *
from collections import Counter
import dashscope
from dashscope.audio.tts_v2 import *

# 配置 API 密钥
api_key = "your-api-key"
os.environ["OPENAI_BASE_URL"] = 'https://api.gpt.ge/v1'
client = OpenAI(api_key=api_key)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 场地参数（基于图像坐标系）
COURT_CENTER = (150, 445)   # 场地中心点 (x,y)
COURT_CENTER_BIAS = (150, 400)  # 场地中心偏移量（x,y）
COURT_WIDTH = 255           # 场地有效宽度（x轴范围22.5~277.5）
COURT_HEIGHT = 590          # 场地有效高度（y轴范围150~740）
LEFT_DOWN = (22.5, 150)      # 场地左下角坐标


# 新增状态管理
chat_history_state = gr.State([])  # 用于存储完整对话历史


# 类别颜色映射
COLOR_MAP = {
    0: "#FF6B6B",  # 红色
    1: "#4ECDC4",  # 青色
    2: "#45B7D1",  # 蓝色
    3: "#96CEB4",  # 绿色
    4: "#FFEEAD",  # 黄色
    5: "#D4A5A5"   # 粉色
}

# 类别名称映射
LABEL_MAP = {
    0: "net shot",
    1: "lift",
    2: "smash",
    3: "defensive drive",
    4: "clear",
    5: "flat shot"
}
def count_reasons(df):
    win_reasons = df['win_reason'].dropna()
    lose_reasons = df['lose_reason'].dropna()

    win_reason_counts = Counter(win_reasons)
    lose_reason_counts = Counter(lose_reasons)

    return win_reason_counts, lose_reason_counts

# 修改 plot_reason_counts 函数以接收 ax 参数
def plot_reason_counts(win_reason_counts, lose_reason_counts, ax=None):
    # 创建画布
    fig = plt.figure(figsize=(12, 6))
    # 创建两个子图
    ax1 = plt.subplot(121) # 左边的子图
    ax2 = plt.subplot(122) # 右边的子图

    # Plotting win_reason counts
    win_reasons, win_counts = zip(*win_reason_counts.items())
    ax1.bar(win_reasons, win_counts, color='lightgreen')
    ax1.set_xlabel('Win Reason')
    ax1.set_ylabel('Count')
    ax1.set_title('Win Reason Counts')
    ax1.tick_params(axis='x', rotation=45)

    # Plotting lose_reason counts
    lose_reasons, lose_counts = zip(*lose_reason_counts.items())
    ax2.bar(lose_reasons, lose_counts, color='lightcoral')
    ax2.set_xlabel('Lose Reason')
    ax2.set_ylabel('Count')
    ax2.set_title('Lose Reason Counts')
    ax2.tick_params(axis='x', rotation=45)

    # 如果 ax 被提供，则将 ax1 和 ax2 的内容绘制到指定的 ax 中
    if ax is not None:
        print("Using provided axes for plotting.")
        for a in [ax1, ax2]:
            for bar in a.containers[0]:
                a.draw_artist(bar)
            a.relim()
            a.autoscale_view()
        fig.canvas.draw_idle()


def visualize_match_data(match_segment):    
    # 划分比赛段落
    if match_segment == "第一场":
        df = pd.read_csv("/home/jovyan/2024-srtp/srtp-final/Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set1.csv")
    elif match_segment == "第二场":
        df = pd.read_csv("/home/jovyan/2024-srtp/srtp-final/Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set2.csv")
    else:
        df = pd.read_csv("/home/jovyan/2024-srtp/srtp-final/Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set3.csv")
    sub_df = df.copy()

    type_mapping = {
        '放小球': 'net shot',
        '擋小球': 'return net',
        '殺球': 'smash',
        '點扣': 'wrist smash',
        '挑球': 'lob',
        '防守回挑': 'defensive return lob',
        '長球': 'clear',
        '平球': 'drive',
        '小平球': 'driven flight',
        '後場抽平球': 'back-court drive',
        '切球': 'drop',
        '過渡切球': 'passive drop',
        '推球': 'push',
        '撲球': 'rush',
        '防守回抽': 'defensive return drive',
        '勾球': 'cross-court net shot',
        '發短球': 'short service',
        '發長球': 'long service'
    }

    # 设置字体支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

    # 创建画布：3行1列，高度增加一点
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(10, 20),gridspec_kw={'height_ratios': [1, 1, 1.5, 1]})
    plt.subplots_adjust(hspace=0.6)  # 增加子图间距

    # 图1：击球类型分布（英文）
    sub_df['type_en'] = sub_df['type'].map(type_mapping)
    type_counts = sub_df['type_en'].value_counts()
    ax1.bar(type_counts.index, type_counts.values, color='skyblue')
    ax1.set_title('Shot Type Distribution')
    ax1.tick_params(axis='x', rotation=90)

    # 图2：击球-落点热力图
    hit_areas = pd.crosstab(sub_df['hit_area'], sub_df['landing_area'])
    im = ax2.imshow(hit_areas, cmap='YlGnBu')
    ax2.set_title('Hit-Landing Area Matrix')
    ax2.set_xlabel('Landing Area')
    ax2.set_ylabel('Hit Area')
    plt.colorbar(im, ax=ax2)
    win_reason_mapping = {
    "對手出界": "Opponent Out of Bounds",
    "對手掛網": "Opponent Netted",
    "對手未過網": "Opponent Failed to Clear the Net",
    "落地致勝": "Winning Shot (Ball Landed)",
    "對手落點判斷失誤": "Opponent Misjudged Landing Spot"
    }
    # 图3：Win Reason Counts（得分原因统计）
    win_reasons_en = sub_df['win_reason'].dropna().map(win_reason_mapping)
    win_reason_counts = Counter(win_reasons_en)

    win_reasons_list, win_counts_list = zip(*win_reason_counts.items())
    ax3.bar(win_reasons_list, win_counts_list, color='lightgreen')
    ax3.set_title('Win Reason Counts')
    ax3.set_xlabel('Win Reason')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    lose_reason_mapping = {
    "對手落地致勝": "Opponent's Winning Shot (Ball Landed)",
    "掛網": "Netted",
    "未過網": "Failed to Clear the Net",
    "出界": "Out of Bounds",
    "落點判斷失誤": "Misjudged Landing Spot"
    }
    # 图4：Lose Reason Counts（失分原因统计）
    lose_reasons = sub_df['lose_reason'].dropna().map(lose_reason_mapping)
    lose_reason_counts = Counter(lose_reasons)
    lose_reasons_list, lose_counts_list = zip(*lose_reason_counts.items())
    ax4.bar(lose_reasons_list, lose_counts_list, color='lightcoral')
    ax4.set_title('Lose Reason Counts')
    ax4.set_xlabel('Lose Reason')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    # 自动调整布局
    plt.tight_layout()

    return fig


def create_timeline_plot(action_list, video_duration):
    """生成时间轴图"""
    # 计算视频时长
    action_list = [1,2,3,4,3,2,5,0,1,3,2,4,5,0,1,2,3,4,5]
    fig, ax = plt.subplots(figsize=(30, 1))
    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 合并连续相同动作
    current_action = action_list[0]
    start_time = 0
    for i in range(1, len(action_list)):
        if action_list[i] != current_action or i == len(action_list)-1:
            end_time = i
            ax.add_patch(Rectangle(
                (start_time, 0), 
                end_time - start_time, 1,
                facecolor=COLOR_MAP[current_action],
                edgecolor='white'
            ))
            start_time = end_time
            current_action = action_list[i]

    # 添加图例
    legend_handles = []
    for label_id, color in COLOR_MAP.items():
        legend_handles.append(
            Rectangle((0,0),1,1, facecolor=color, label=LABEL_MAP[label_id])
        )
    ax.legend(handles=legend_handles, 
             loc='upper center',
             bbox_to_anchor=(0.5, -0.2),
             ncol=6)

    return fig


# 设置 DashScope API Key 和模型参数
dashscope.api_key = "sk-cd5c2f5fcddd49c5b4e4169d5021d8e2"
TTS_MODEL = "cosyvoice-v1"

def commentary_generator(rally_id, chat_response, rally_data,voice_choice):
    # 路径配置
    rally_id = 3
    rally_data = rally_data["rally"]
    rally= rally_data[rally_id-1]
    start_frame = int(rally[0])
    end_frame = int(rally[1])
    base_dir = "/home/jovyan/2024-srtp/srtp-final"
    json_path = os.path.join(base_dir, "hit_frame_detection", "outputs", "joints", "input_video", f"rally_{rally_id}.json")
    video_input_path = os.path.join(base_dir, "hit_frame_detection", "outputs", "videos", "input_video", "video_1_h264.mp4")
    video_output_path = os.path.join(base_dir, "output_videos", f"video_{rally_id}_with_commentary.mp4")

    # 创建输出目录
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    hit_frames = data.get("hit frames", [])
    print(f"加载到 {len(hit_frames)} 个击球时间点")

    # 解析 chat_response（格式：[{"hit_num": 2, "comment": "..."}, ...]）
    try:
        commentaries = json.loads(chat_response)
    except json.JSONDecodeError:
        raise ValueError("解说词格式错误，请检查输入的 JSON 数据！")

    # 检查格式是否正确
    if not isinstance(commentaries, list) or not all(
        isinstance(item, dict) and "hit_num" in item and "comment" in item for item in commentaries
    ):
        raise ValueError("解说词格式错误，请确保每个解说词包含 'hit_num' 和 'comment' 字段！")

    # 加载原始视频
    video = VideoFileClip(video_input_path)

    # 初始化临时文件夹用于存储音频
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_clips = []

        # 初始化合成器
        synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=voice_choice)

        # 遍历每一条解说词
        for i, item in enumerate(commentaries):
            hit_num = item["hit_num"]
            comment = item["comment"]

            # 根据 hit_num 获取对应的帧编号
            if hit_num < 1 or hit_num > len(hit_frames):
                raise ValueError(f"无效的 hit_num: {hit_num}，请检查解说词中的 hit_num 是否在有效范围内！")

            frame_time = hit_frames[hit_num - 1]  # hit_num 是从 1 开始计数的

            # 帧转秒（假设帧率是 30 fps）
            second = frame_time / 30.0
            print(f"第{i+1}句解说：在 {second:.2f}s 处插入，hit_num: {hit_num}")

            # 生成 TTS 音频
            output_audio_path = os.path.join(tmpdir, f"tts_{i}.mp3")
            audio_data = synthesizer.call(comment)
            with open(output_audio_path, 'wb') as f:
                f.write(audio_data)

            # 加载音频片段并设置开始时间
            audio_clip = AudioFileClip(output_audio_path).set_start(second)
            audio_clips.append(audio_clip)

            # 防止并发请求限制
            time.sleep(1)

        # 合成原视频音频 + 解说音频
        final_audio = CompositeAudioClip([video.audio] + audio_clips) if video.audio else CompositeAudioClip(audio_clips)

        # 设置视频音频并导出
        video_with_audio = video.set_audio(final_audio)
        video_with_audio.write_videofile(video_output_path, codec="libx264", audio_codec="aac")

        print(f"视频已保存至: {video_output_path}")


def get_action_plot(rally_choice,rally_data):
    
    video_path_part = rally_choice.split("/")[-1].split(".mp4")[0]
    # 提取rally_id：
    rally_id = int(video_path_part.split("_")[-1])
    # 生成时间轴图
    video_path_part = rally_choice.split("/")[-1].split(".mp4")[0]
    # 提取rally_id：
    rally_id = int(video_path_part.split("_")[-1])
    pred_list = get_actions(rally_id)
    # 提取rally_data中的rally_id对应的帧范围
    rally_data = rally_data["rally"]
    rally= rally_data[rally_id-1]
    start_frame = int(rally[0])
    end_frame = int(rally[1])
    duration = (end_frame - start_frame) / 30  # 假设帧率为30fps
    fig = create_timeline_plot(pred_list, duration)
    return fig


def get_actions(rally_id):
      #从../joints/中读取csv文件
    action_cmd = f"bash /home/jovyan/2024-srtp/srtp-final/mmaction2/get_action.sh {rally_id}"
    subprocess.run(action_cmd, shell=True, check=True)
    json_path = "/home/jovyan/2024-srtp/srtp-final/mmaction2/output_with_names.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    pred_list = []
    label_names=[
    "net shot",
    "lift",
    "smash",
    "defensive drive",
    "clear",
    "flat shot"
    ]
    for entry in data:
        pred_label = entry['pred_label'][0]
        pred_label_name = label_names[pred_label]
        pred_list.append(pred_label)
    print(pred_list)
    return pred_list
# 修改后的预测函数
def qwen_predict(user_input, sport, player1_name, player2_name,rally_data,rally_choice):
    # 原有预测逻辑（假设返回字符串）
    # 解析路径：video_{rally_id}_h264.mp4,提取rally_id
    video_path_part = rally_choice.split("/")[-1].split(".mp4")[0]
    # 提取rally_id：
    rally_id = int(video_path_part.split("_")[-1])
    # 提取rally_data中的rally_id对应的帧范围
    rally_data = rally_data["rally"]
    rally= rally_data[rally_id-1]
    start_frame = int(rally[0])
    end_frame = int(rally[1])
    # 获取动作序列
    action_list = get_actions(rally_id)
    player_nickname_map = {
    "林丹": "丹",
    "李宗伟": "李",
    "谌龙": "龙",
    "桃田贤斗": "桃",
    "安赛龙": "龙",
    "石宇奇": "石头",
    "乔纳坦·克里斯蒂": "乔",
    "李梓嘉": "李",
    "安东尼·西尼苏卡·金廷": "金廷",
    "周天成": "周"
    }
    player1_nickname = player_nickname_map.get(player1_name, player1_name)
    player2_nickname = player_nickname_map.get(player2_name, player2_name)  
    prompt = f"""
        Generate a professional badminton commentary in Chinese for a {sport} match between {player1_name} (Player 1) and {player2_name} (Player 2). 

        Key requirements:
        1. Output Format:
        Return only a valid JSON array of objects with format, where hit_num is the index of the hit in the rally (1-based), and comment is the generated commentary for that hit:
        [
        {
            "hit_num": 2,
            "comment": "..."
        }
        {
            "hit_num": 3,
            "comment": "..."
        },
        ...
        ]
        

        2. Identify and classify each shot type from these 20 categories:
        [net shot, smash, wrist smash, lob, defensive return lob, clear, drive, 
        driven flight, back-court drive, drop, passive drop, push, rush, 
        defensive return drive, cross-court net shot, short service, 
        long service, defensive shot, push/rush]

        3. Each comment sentence should include:
        - Name of the player who hit the ball in short: the last name of the player, e.g. "李" for "李宗伟"
        - Player action sequence (with shot types)
        - Tactical analysis
        - Score update(get the match score from left-upper corner of the video)
        - Exciting/exclamatory commentary phrase

        4. Style guidelines:
        - Use nicknames or abbreviations like “{player1_nickname}” and “{player2_nickname}” when appropriate.
        - Chinese commentary only
        - Short, dynamic sentences (3 to 7 words for action descriptions)
        - Tactical insights e.g. ("控网抢攻", "逼压底线")
        - Do not generate too much emotional highlights

        Generate commentary that:
        1. Precisely identifies each shot type
        2. Explains player strategies
        3. Maintains exciting play-by-play flow
        4. Updates score regularly
        5. Avoids excessive emotional highlights
        6. Only Chinese commentary
        extra requirements by user:
        {user_input}
        """
    print(prompt)
    prediction = get_qwen_ans(rally_choice, prompt, model, tokenizer, max_num_frames, generation_config) 
    return prediction

# 对话历史更新函数
def update_chat_history(video_input, user_input, history):
    # 获取模型响应
    model_response = qwen_predict(video_input, user_input)
    
    # 构建新的对话条目（匹配图像中的消息格式）
    new_entry = [
        {"role": "user", "content": f"📥 输入：{user_input}"},
        {"role": "assistant", "content": f"⚡ 响应：{model_response}"}
    ]
    
    # 更新历史记录
    updated_history = history + new_entry if history else new_entry
    
    # 保留最新5条对话（防止溢出）
    return updated_history[-10:]

def coordinate_transform(df):
    """坐标变换流程：中心对齐→缩放"""
    # 原始数据矩阵（示例字段）
    df = df.dropna(subset=['hit_x', 'hit_y', 
                            'landing_x', 'landing_y',
                            'player_location_x', 'player_location_y',
                            'opponent_location_x', 'opponent_location_y'])
    hit_points = df[['hit_x', 'hit_y']].values
    landing_points = df[['landing_x', 'landing_y']].values
    player_locations = df[['player_location_x', 'player_location_y']].values
    opponent_locations = df[['opponent_location_x', 'opponent_location_y']].values
    print(hit_points.shape, landing_points.shape, player_locations.shape, opponent_locations.shape)
    # 计算x,y坐标的均值（数据中心）
    all_points = np.vstack((hit_points, landing_points, player_locations, opponent_locations))
    print(all_points.shape)
    data_center = np.mean(all_points, axis=0)
    
    
    # Step3: 平移对齐场地中心
    offset = np.array(COURT_CENTER) - data_center
    print(offset)
    aligned = all_points + offset
    
    # Step4: 自适应缩放
    # 计算数据范围
    x_min, x_max = np.min(aligned[:,0]), np.max(aligned[:,0])
    y_min, y_max = np.min(aligned[:,1]), np.max(aligned[:,1])
    
    # 计算缩放比例（保留5%边界）
    scale_x = COURT_WIDTH * 1.1 / (x_max - x_min)
    scale_y = COURT_HEIGHT * 1.1 / (y_max - y_min)
    
    rotation_matrix = np.array([[scale_x, 0],
                                [0, scale_y]])
    print("scale_x, scale_y", scale_x, scale_y)

    hit_points_transformed = np.dot(hit_points, rotation_matrix) 
    landing_points_transformed = np.dot(landing_points, rotation_matrix)
    player_locations_transformed = np.dot(player_locations, rotation_matrix) 
    opponent_locations_transformed = np.dot(opponent_locations, rotation_matrix) 
    all_points_transformed = np.vstack((hit_points_transformed, landing_points_transformed, player_locations_transformed, opponent_locations_transformed))
    data_center_transformed = np.mean(all_points_transformed, axis=0)
    print("data_center_transformed", data_center_transformed)
    # 计算新的偏移量
    offset_transformed = np.array(COURT_CENTER_BIAS) - data_center_transformed
    print("offset_transformed", offset_transformed)
    # 平移对齐
    hit_points_transformed += offset_transformed
    landing_points_transformed += offset_transformed
    player_locations_transformed += offset_transformed
    opponent_locations_transformed += offset_transformed

    # 更新 DataFrame
    df[['hit_x', 'hit_y']] = hit_points_transformed
    df[['landing_x', 'landing_y']] = landing_points_transformed
    df[['player_location_x', 'player_location_y']] = player_locations_transformed
    df[['opponent_location_x', 'opponent_location_y']] = opponent_locations_transformed

    return df



def draw_heatmap(ax, df, player):
    """绘制分层热力饼图"""
    player_df = df[(df['player'] == player) & 
                  (df['landing_area'] <= 10)]
    
    # 统计区域击球频次
    area_counts = player_df['landing_area'].value_counts()
    max_count = area_counts.max() if not area_counts.empty else 1
    
    # 设置渐变色
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(area_counts)))
    
    for (area, count), color in zip(area_counts.items(), colors):
        if area not in AREA_CENTERS:
            continue
            
        # 调整垂直位置
        x, y = AREA_CENTERS[area]
        if player == 'B':
            y = COURT_HEIGHT - y  # 下半部分镜像
        
        # 动态计算半径和透明度
        radius = 40 + (count / max_count) * 80
        alpha = 0.4 + (count / max_count) * 0.4
        
        ax.add_patch(Circle(
            (x, y), radius=radius,
            facecolor=color, edgecolor='white',
            linewidth=0.8, alpha=alpha
        ))
        
        # 添加频次标注
        ax.text(x, y, str(count), 
               ha='center', va='center',
               color='white', fontsize=10, 
               fontweight='bold')


def draw_match_data(mode='all'):
    # 复用之前的场地绘制函数
    fig = draw_badminton_court()  
    ax = fig.gca()
    
    # 读取比赛数据
    df = pd.read_csv("/home/jovyan/2024-srtp/srtp-final/Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set1.csv")
    df = coordinate_transform(df)  # 执行坐标变换
    
    # 可视化参数设置
    plot_config = {
        'hit_point': {'color': 'red', 'marker': 'o', 's': 80},
        'landing_point': {'color': 'blue', 'marker': 'X', 's': 100},
        'player_A': {'color': 'lime', 'marker': 'P', 's': 120},
        'player_B': {'color': 'yellow', 'marker': 'D', 's': 120},
        'trajectory': {'color': 'white', 'linestyle': '-', 'linewidth': 0.3}
    }

    for _, row in df.iterrows():
        if(mode == 'ball' or mode == 'all'):
            # 击球轨迹可视化
            ax.plot([row['hit_x'], row['landing_x']],
                    [row['hit_y'], row['landing_y']],
                    **plot_config['trajectory'])
            # 击球点标记
            ax.scatter(row['hit_x'], row['hit_y'],
                    label='hit', **plot_config['hit_point'])
            ax.scatter(row['landing_x'], row['landing_y'],
                    label='land', **plot_config['landing_point'])
            ax.add_patch(Circle((row['landing_x'], row['landing_y']), 
                            radius=1, fc='none', ec='cyan', lw=0.3))

        
        if(mode == 'player' or mode == 'all'):
            player_color = plot_config['player_A'] if row['player'] == 'A' else plot_config['player_B']
            ax.scatter(row['player_location_x'], row['player_location_y'],
                    label=f'player {row["player"]}', **player_color)
            ax.scatter(row['opponent_location_x'], row['opponent_location_y'],
                    label=f'player {row["player"]}', **player_color)
       

    # 添加智能图例（自动去重）
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # 去重
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc='upper left', fontsize=8, ncol=2)

    return fig


def draw_badminton_court():
    fig, ax = plt.subplots(figsize=(6, 12), dpi=80)
    fig.set_facecolor((0/255, 127/255, 102/255))  # 设置背景颜色
    ax.set_facecolor((0/255, 127/255, 102/255))
    
    # 设置坐标范围（单位：厘米）
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制外边框
    outer = Rectangle((22.5, 150), 255, 590, linewidth=4, 
                     edgecolor='white', facecolor='none')
    ax.add_patch(outer)

    # 绘制内边框
    # inner = Rectangle((45, 195), 210, 500, linewidth=2,
    #                  edgecolor='white', facecolor='none')
    # ax.add_patch(inner)

    #把内边框改成四条线，即原来内边框的延申
    #各个参数的意义；
    # [x1, x2]：线段的起始和结束位置
    # [y1, y2]：线段的起始和结束位置
    ax.plot([22.5, 275.5], [195, 195], color='white', linewidth=2)  # 上边线
    ax.plot([22.5, 275.5], [695, 695], color='white', linewidth=2)  # 下边线
    ax.plot([45, 45], [150, 695+45], color='white', linewidth=2)  # 左边线
    ax.plot([255, 255], [150, 695+45], color='white', linewidth=2)  # 右边线
    

    # 绘制中线
    ax.plot([22.5, 277.5], [445, 445], color='white', linewidth=2)
    ax.plot([150, 150], [150, 150+295/3*2], color='white', linewidth=2)
    ax.plot([150, 150], [740-295/3*2, 740], color='white', linewidth=2)

    # 绘制发球线
    for y in [150+295/3*2, 740-295/3*2]:
        ax.plot([22.5, 277.5], [y, y], color='white', linewidth=2)

    # 绘制网格线（横向）
    for y in [150 + 295/3 * i for i in range(1,2)]:
        ax.plot([22.5, 277.5], [y, y], color='#25cf98', linewidth=1)

    for y in [740-295/3 * i for i in range(1,2)]:
        ax.plot([22.5, 277.5], [y, y], color='#25cf98', linewidth=1)

    # 绘制网格线（纵向）
    for x in [22.5 + 255/3 * i for i in [1,2]]:
        ax.plot([x, x], [150, 740], color='#25cf98', linewidth=1)

    # 添加文字标签
    positions = [
        (72.5, 150+295/6*5, "1"), (157.5, 150+295/6*5, "2"), (242.5, 150+295/6*5, "3"),
        (72.5, 150+295/2, "4"), (157.5, 150+295/2, "5"), (242.5, 150+295/2, "6"),
        (72.5, 150+295/6+10, "7"), (157.5, 150+295/6+10, "8"), (242.5, 150+295/6+10, "9"),
        (72.5, 740-295/6*5, "3"), (157.5, 740-295/6*5, "2"), (242.5, 740-295/6*5, "1"),
        (72.5, 740-295/2, "6"), (157.5, 740-295/2, "5"), (242.5, 740-295/2, "4"),
        (72.5, 740-295/6-10, "9"), (157.5, 740-295/6-10, "8"), (242.5, 740-295/6-10, "7")
    ]
    
    for x, y, text in positions:
        ax.text(x, y, text, color='#25cf98', ha='center', va='center', fontsize=20)

    return fig


def ragflow_predict(chat_response, sport, player1_name, player2_name):
    # 供 Gradio 调用的接口函数
# def query_rag(question: str) -> str:
#     return rag_service.query(question)
    prompt = f"""
    Generate a professional badminton commentary in Chinese for a {sport} match between {player1_name} (Player 1) and {player2_name} (Player 2).
    Here is the initially generated commentary:
    {chat_response}
    Key requirements:
    1. Output Format:
    Return only a valid JSON array of objects with format, where hit_num is the index of the hit in the rally (1-based), and comment is the generated commentary for that hit:
    [
    {
        "hit_num": 2,
        "comment": "..."
    }
    {
        "hit_num": 3,
        "comment": "..."
    },
    ...
    ]
    2. Inhance the commentary by:
    - base on the initial commentary
    - base on the database knowledge of badminton commentary
    - Adding more tactical analysis
    - Including more player actions
    - Making it more exciting
    3. Style guidelines:
    - Use nicknames or abbreviations like “{player1_name}” and “{player2_name}” when appropriate.
    - Chinese commentary only
    - Short, dynamic sentences (3 to 7 words for action descriptions)
    - Tactical insights e.g. ("控网抢攻", "逼压底线")
    """
    ans = query_rag(prompt)
    return ans


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=history_openai_format,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

def process_video(video_file, sport, structured_text):
    video_file.save_to_disk("temp_video.mp4")  # Save the uploaded video to disk
    output = {
        "video_processed": True,
        "sport": sport,
        "structured_text": structured_text,
        "message": "Video processing completed!"
    }
    return json.dumps(output, indent=4)

def get_today():
    import datetime
    file_path_vid =  datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = f"data/{file_path_vid}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return file_path_vid
    
def segment_video(video_input):
    if video_input is None:
        return "No video input provided.", None
    
    # mv outputs outputs_old
    # rm -rf ./videos
    # mkdir videos
    # cp ../data/tennis/test_videos/$VIDEO.mp4 ./videos
    # echo "output cleaned"
    # mkdir outputs
    # cd src
    # python main.py

    # 1. 创建必要的目录结构
    base_dir = Path("hit_frame_detection")
    videos_dir = base_dir / "videos"
    outputs_dir = base_dir / "outputs"
    outputs_old_dir = base_dir / "outputs_old"
    
    try:
        # 2. 清理旧输出并创建新目录
        # if outputs_dir.exists():
        #     if outputs_old_dir.exists():
        #         shutil.rmtree(outputs_old_dir)
        #     shutil.move(outputs_dir, outputs_old_dir)
        
        # if not videos_dir.exists():
        #     videos_dir.mkdir(parents=True)
        
        # outputs_dir.mkdir(exist_ok=True)
        
        # 3. 保存上传的视频文件
        video_path = videos_dir / "input_video.mp4"
        # 直接cp到video_path：
        shutil.copy(video_input, video_path)
        
        # 4. 执行处理脚本
        src_dir = base_dir / "src"
        subprocess.run(
            ["python", "main.py"],
            cwd=src_dir,
            check=True
        )
        print( "Video processing completed successfully.")
        # 5. 处理结果并返回
        gallery_items, rally_data, video_info = process_video_result(outputs_dir)
    
        # 构造Dropdown的选项：格式为[("显示标签", "实际值"), ...]
        dropdown_choices = [(f"回合 {rid}", path) for rid, path in video_info]
        
        # 返回更新组件
        rally_choices = gr.Dropdown(choices=dropdown_choices, visible=True)
        return gallery_items, rally_data, rally_choices
        
    except subprocess.CalledProcessError as e:
        print( f"Error during video processing: {e.stderr}")
        error_msg = f"Error processing video: {e.stderr}"
        return error_msg, None
    except Exception as e:
        print( f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}", None


def process_video_result(outputs_dir):
    """处理输出目录中的结果"""
    # 读取回合JSON文件
    rally_json_path = Path(outputs_dir) / "rallies" / "input_video.json"
    with open(rally_json_path, 'r') as f:
        rally_data = json.load(f)
    
    # 获取视频输出目录
    video_output_dir = Path(outputs_dir) / "videos" / "input_video"

    # 构建Gallery兼容格式
    gallery_items = []
    # [(f"回合 {rid}", path) for rid, path in video_info]
    video_info = []
    for rally in rally_data["rally"]:
        start_frame, end_frame, rally_id = rally
        video_path = video_output_dir / f"video_{rally_id}.mp4"
        output_video_path = video_output_dir / f"video_{rally_id}_h264.mp4"
        # ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4
        # 这里使用ffmpeg将视频文件转换为h264编码格式
        # ffmpeg_cmd = f"ffmpeg -y -i {video_path} -c:v libx264 -c:a copy {output_video_path}"
        # subprocess.run(ffmpeg_cmd, shell=True, check=True)
        # 检查视频文件是否存在
        
        # Gradio Gallery支持两种格式：
        # 1. 纯文件路径字符串
        # 2. (文件路径, 标题) 的元组
        if not video_path.exists():
            print(f"Warning: Video file {video_path} does not exist.")
            continue
        # 加载视频文件:
        # from video_path：

        print (f"Rally {rally_id}: {video_path}, frames {start_frame}-{end_frame}")
        print(str(video_path))
        video_info.append((rally_id, str(video_path)))
        gallery_items.append((
            str(output_video_path),  # 视频文件路径
            f"回合 {rally_id}\n帧范围: {start_frame}-{end_frame}"  # 显示标题
        ))
        
    
    return gallery_items,rally_data,video_info

with gr.Blocks() as demo:
    gr.Markdown("""
        <h1 style='text-align: center;'>基于视觉大模型的体育智能分析</h1>
        <div style='display: flex; justify-content: center; gap: 20px;'>
            <h3 style='text-align: center; color: gray'>2025-srtp jyHu syHu rhXu</h3>
            <a href="https://github.com/opengvlab/VideoChat-Flash-Qwen2_5-2B_res448" target="_blank">
                <img src="https://raw.githubusercontent.com/little612pea/srtp-2025/main/logos/gvlab.jpg" alt="VideoChat-Flash-Qwen2_5-2B_res448" width="50">
            </a>
            <a href="https://github.com/open-mmlab/mmaction2" target="_blank">
                <img src="https://raw.githubusercontent.com/little612pea/srtp-2025/main/logos/mmaction2_logo.png" alt="mmaction2" width="140">
            </a>
            <a href="URL_TO_RAGFLOW" target="_blank">
                <img src="https://raw.githubusercontent.com/little612pea/srtp-2025/main/logos/ragflow-logo.png" alt="ragflow" width="130">
            </a>
            <a href="https://www.bwfbadminton.com/" target="_blank">
                <img src="https://raw.githubusercontent.com/little612pea/srtp-2025/main/logos/BWF-LOGO.jpg" alt="BWF" width="50">
            </a>
            <a href="https://www.itftennis.com/en/" target="_blank">
                <img src="https://raw.githubusercontent.com/little612pea/srtp-2025/main/logos/itf.jpg" alt="ITF" width="121">
            </a>
        </div>
        """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Collect Info From Video")

            video_input = gr.Video(label="上传视频文件")
            video_input.GRADIO_CACHE = f"data/{get_today()}"
            segment_button = gr.Button("分割回合", variant="primary")
            rally_data = gr.JSON(
                label="分割结果",
                visible=False  # 确保组件可见
            )
        

            sport_dropdown = gr.Dropdown(
                choices=["乒乓球", "羽毛球", "网球"],
                label="选择体育运动",
                value="羽毛球"
            )
            players_options = ["林丹", "李宗伟", "谌龙", "桃田贤斗", "安赛龙", "石宇奇", "乔纳坦·克里斯蒂", "李梓嘉", "安东尼·西尼苏卡·金廷", "周天成"]

            player1_name = gr.Dropdown(
                choices=players_options,
                label="Player 1",
                value="谌龙"  # Set default selected player
            )

            player2_name = gr.Dropdown(
                choices=players_options,
                label="Player 2",
                value="林丹"  # Set default selected player
            )
            video_dropdown = gr.Dropdown(
                label="选择回合视频",
                choices=[1,2],  # 初始为空
                visible=True,  # 初始隐藏，处理完成后显示
                value=1
           )
        with gr.Column(scale=6,min_width=700):
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            , columns=[3], rows=[1], object_fit="contain", height="auto")
            plot_video_action_list = gr.Plot(label="action sequence")
            #上传csv文件:
            vis_choice =  gr.Dropdown(choices=["player", "ball", "all"], value="player")
            with gr.Column():
                actions_button = gr.Button("Get Action", variant="primary")
                upload_csv_button = gr.Button("Get Match Statistics", variant="primary")
           

            #控制可见的图例:


        with gr.Column(scale=4):
            match_segment = gr.Dropdown(
                choices=["第一场", "第二场", "第三场"],
                label="Select Match Segment",
                value="第一场"
            )
             
            # 新增可视化按钮
            stats_button = gr.Button("Show Match Stats", variant="primary")
            
            # 新增统计图表展示区域
            stats_plot = gr.Plot(label="Match Statistics")
            gr.Markdown("## Match Statistics Visualization")
            plot = gr.Plot(label="Badminton Court",value=draw_badminton_court())
    with gr.Row():
        gr.Markdown("## Build Commentary Agent")    
        with gr.Column():
            gr.Markdown("### 解说生成")  
            chat_response = gr.Textbox(
                label="AI Commentary",
                placeholder="AI Commentary",
                lines=10
            )
            user_input = gr.Textbox(label="用户输入", lines=3)
            qwen_button = gr.Button("执行分析", variant="primary")
                
        with gr.Column():
            gr.Markdown("### ragflow解说增强")
            rag_flow_button = gr.Button("Ragflow", variant="primary")
            
        with gr.Column():
            gr.Markdown("### tts语音合成")
            comment_res = gr.Video(label="video will be shown here")
            voice_choice =  gr.Dropdown(choices=["longlaotie", "longcheng", "longhua"], value="longlaotie")
            voice_button = gr.Button("语音合成", variant="primary")


 # 交互逻辑绑定

    rag_flow_button.click(
        fn=ragflow_predict,
        inputs=[chat_response, sport_dropdown, player1_name, player2_name],
        outputs=chat_response
    )
    voice_button.click(
        fn=commentary_generator,
        inputs=[video_dropdown,chat_response,rally_data],
        outputs=None
    )
    actions_button.click(
        fn=get_action_plot,
        inputs=[video_dropdown,rally_data],
        outputs=plot_video_action_list
    )
    stats_button.click(
        fn=visualize_match_data,
        inputs=[match_segment],
        outputs=stats_plot
    )
    qwen_button.click(
        fn=qwen_predict,
        inputs=[user_input, sport_dropdown, player1_name, player2_name,rally_data,video_dropdown],
        outputs=chat_response
    )
    upload_csv_button.click(
        fn=draw_match_data,
        inputs=[vis_choice],
        outputs=plot
    )
    segment_button.click(
        fn=segment_video,
        inputs=video_input,
        outputs=[gallery,rally_data,video_dropdown]
    )

if __name__ == "__main__":
    # model, tokenizer, image_processor, max_num_frames, generation_config = get_qwen_model()
    demo.launch(share=True)
