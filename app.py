from temp_qwen import *
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

# 修改后的预测函数
def qwen_predict(user_input, sport, player1_name, player2_name,rally_data,rally_choice):
    # 原有预测逻辑（假设返回字符串）
    # 解析路径：video_{rally_id}_h264.mp4,提取rally_id
    video_path_part = rally_choice.split("/")[-1].split(".mp4")[0]
    # 提取rally_id：
    rally_id = int(video_path_part.split("_")[-1])
    # 提取rally_data中的rally_id对应的帧范围
    print(rally_data)
    rally_data = rally_data["rally"]
    rally= rally_data[rally_id-1]
    start_frame = int(rally[0])
    end_frame = int(rally[1])
    prompt = f"""
        Generate professional badminton commentary for a {sport} match between {player1_name} (Player 1) and {player2_name} (Player 2). 

        Key requirements:
        1. Identify and classify each shot type from these 20 categories:
        [net shot, smash, wrist smash, lob, defensive return lob, clear, drive, 
        driven flight, back-court drive, drop, passive drop, push, rush, 
        defensive return drive, cross-court net shot, short service, 
        long service, defensive shot, push/rush]

        2. Follow this commentary structure for each rally:
        - Timecode: [start_time] --> [end_time]
        - Player action sequence (with shot types)
        - Tactical analysis
        - Score update
        - Exciting/exclamatory commentary phrase

        3. Style guidelines:
        - Bilingual mix (Chinese technical terms + English phrases)
        - Short, dynamic sentences (3-7 words for action descriptions)
        - Tactical insights ("控网抢攻", "逼压底线")
        - Emotional highlights ("Beautiful shot!", "What a rally!")
        - Current score after each rally ("七比三")

        Example format:
        00:00:02,060 --> 00:00:02,740
        {player1_name} serves with a short service
        {player2_name} returns with cross-court net shot
        "Nice placement!" 

        00:00:02,740 --> 00:00:03,779
        {player1_name} counters with rushing net shot
        {player2_name} lifts defensive lob
        "Great defensive play!"

        The current video is the {rally_id}'th rally from the match, 
        which lasts from frame {start_frame} to frame {end_frame}.

        Generate commentary that:
        1. Precisely identifies each shot type
        2. Explains player strategies
        3. Maintains exciting play-by-play flow
        4. Updates score regularly
        5. Uses natural bilingual expressions
        extra requirements by user:
        {user_input}
        """

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


def draw_match_data(csv_path,mode='all'):
    # 复用之前的场地绘制函数
    fig = draw_badminton_court()  
    ax = fig.gca()
    
    # 读取比赛数据
    df = pd.read_csv(csv_path)
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
        ffmpeg_cmd = f"ffmpeg -y -i {video_path} -c:v libx264 -c:a copy {output_video_path}"
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
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
        <h3 style='text-align: center; color: gray'>2024-srtp jyHu syHu rhXu</h3>
        """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Collect Info From Video")

            video_input = gr.Video(label="上传视频文件")
            video_input.GRADIO_CACHE = f"data/{get_today()}"
            segment_button = gr.Button("分割回合")
            rally_data = gr.JSON(
                label="分割结果",
                visible=True  # 确保组件可见
            )
        

            sport_dropdown = gr.Dropdown(
                choices=["乒乓球", "羽毛球", "网球"],
                label="选择体育运动",
                value="羽毛球"
            )

            players_options = ["Chen Long", "Lin Dan", "Lee Chong Wei", "Kento Momota", "Viktor Axelsen", "Shi Yuqi", "Jonatan Christie", "Ng Ka Long Angus", "Kidambi Srikanth", "Chou Tien-chen"]

            player1_name = gr.Dropdown(
                choices=players_options,
                label="Player 1",
                value="Chen Long"  # Set default selected player
            )

            player2_name = gr.Dropdown(
                choices=players_options,
                label="Player 2",
                value="Lin Dan"  # Set default selected player
            )
        with gr.Column(scale=6,min_width=700):
            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            , columns=[3], rows=[1], object_fit="contain", height="auto")
               
            #上传csv文件:
            csv_file_upload = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            csv_file_upload.GRADIO_CACHE = f"data/{get_today()}"
            vis_choice =  gr.Dropdown(choices=["player", "ball", "all"], value="player")
            upload_csv_button = gr.Button("Upload CSV")
            #控制可见的图例:


        with gr.Column(scale=2):
            gr.Markdown("## Match Statistics Visualization")
            plot = gr.Plot(label="Badminton Court",value=draw_badminton_court())
    with gr.Row():
        gr.Markdown("## Build Commentary Agent")        
        chat_response = gr.Textbox(
            label="AI Commentary",
            placeholder="AI Commentary",
            lines=10
        )
        user_input = gr.Textbox(label="用户输入", lines=3)
        video_dropdown = gr.Dropdown(
            label="选择回合视频",
            choices=[1,2],  # 初始为空
            visible=True,  # 初始隐藏，处理完成后显示
            value=1
        )
        qwen_button = gr.Button("执行分析", variant="primary")
            

 # 交互逻辑绑定
    qwen_button.click(
        fn=qwen_predict,
        inputs=[user_input, sport_dropdown, player1_name, player2_name,rally_data,video_dropdown],
        outputs=chat_response
    )
    upload_csv_button.click(
        fn=draw_match_data,
        inputs=[csv_file_upload, vis_choice],
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
