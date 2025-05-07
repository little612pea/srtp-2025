import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.6)  # 增加子图间距
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', which='both', bottom=False)  # 隐藏x轴的刻度线
    # 图1：击球类型分布（英文）
    sub_df['type_en'] = sub_df['type'].map(type_mapping)
    type_counts = sub_df['type_en'].value_counts()

    # 为每个柱状图分配一个特定的颜色
    colors = plt.cm.get_cmap('tab20', len(type_counts)).colors

    bars = ax1.bar(type_counts.index, type_counts.values, color=colors)
    ax1.set_title('Shot Type Distribution')
    ax1.tick_params(axis='x', rotation=90)

    # 设置图例到最佳位置，比如右上角，并分成两列
    ax1.legend(bars, type_counts.index, bbox_to_anchor=(1.1, 1), loc='upper right', ncol=2)

    # 如果需要进一步微调图例的位置，可以调整 bbox_to_anchor 的坐标值
    # 例如: bbox_to_anchor=(1.05, 1) 或其他适合的值

    # 图2：击球-落点热力图
    # 先去除landing area>9的值：
    sub_df = sub_df[sub_df['landing_area'] <= 9]
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
    "落地致勝": "Ball Landed",
    "對手落點判斷失誤": "Opponent Misjudged Landing Spot"
    }
    # 图3：Win Reason Counts（得分原因统计）
    win_reasons_en = sub_df['win_reason'].dropna().map(win_reason_mapping)
    win_reason_counts = Counter(win_reasons_en)

    win_reasons_list, win_counts_list = zip(*win_reason_counts.items())

    # 为每个柱状图分配一个特定的颜色
    colors = plt.cm.get_cmap('tab10', len(win_reasons_list)).colors
    ax3.set_xticklabels([])
    ax3.tick_params(axis='x', which='both', bottom=False)  # 隐藏x轴的刻度线
    ax3.bar(win_reasons_list, win_counts_list, color=colors)
    ax3.set_title('Win Reason Counts')
    ax3.set_xlabel('Win Reason')
    ax3.set_ylabel('Count')

    # 设置图例到最佳位置，比如右上角
    ax3.legend(ax3.patches, win_reasons_list, bbox_to_anchor=(1.1, 1), loc='upper right')
    lose_reason_mapping = {
    "對手落地致勝": "Opponent's Ball Landed",
    "掛網": "Netted",
    "未過網": "Failed to Clear the Net",
    "出界": "Out of Bounds",
    "落點判斷失誤": "Misjudged Landing Spot"
    }
    # 图4：Lose Reason Counts（失分原因统计）
    lose_reasons = sub_df['lose_reason'].dropna().map(lose_reason_mapping)
    lose_reason_counts = Counter(lose_reasons)
    lose_reasons_list, lose_counts_list = zip(*lose_reason_counts.items())

    # 为每个柱状图分配一个特定的颜色
    colors = plt.cm.get_cmap('Paired', len(lose_reasons_list)).colors
    # 去除底部的x轴标注
    ax4.set_xticklabels([])
    ax4.tick_params(axis='x', which='both', bottom=False)  # 隐藏x轴的刻度线
    ax4.bar(lose_reasons_list, lose_counts_list, color=colors)
    ax4.set_title('Lose Reason Counts')
    ax4.set_xlabel('Lose Reason')
    ax4.set_ylabel('Count')

    # 设置图例到最佳位置，比如右上角
    ax4.legend(ax4.patches, lose_reasons_list, bbox_to_anchor=(1.1, 1), loc='upper right')

    # 自动调整布局
    plt.tight_layout()

    return fig

if __name__ == "__main__":
    match_segment = "第三场"  # 替换为你想要的比赛段落
    fig = visualize_match_data(match_segment)
    fig.savefig(f"{match_segment}_match_data.png")  # 保存图形
    print(f"图形已保存为 {match_segment}_match_data.png")