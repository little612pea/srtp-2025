import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

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

def create_heatmap():
    # 这里可以添加热力图生成逻辑
    fig = draw_badminton_court()
    return fig

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            plot = gr.Plot(label="Badminton Court")
            update_btn = gr.Button("Update Visualization")
            
        with gr.Column():
            gr.Markdown("### Control Panel")
            player_a_pos = gr.Slider(0, 295, label="Player A X Position")
            player_b_pos = gr.Slider(0, 660, label="Player B Y Position")
            
    update_btn.click(
        fn=create_heatmap,
        outputs=plot
    )

if __name__ == "__main__":
    demo.launch()