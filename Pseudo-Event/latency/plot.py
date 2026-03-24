import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# --- 1. 定义路径和阶段名称 ---
files = {
    'Stage 1: Static': 'performance_log1.txt',
    'Stage 2: Camera Motion': 'performance_log2.txt',
    'Stage 3: Dynamic Object': 'performance_log3.txt'
}

def parse_log_robust(file_path):
    """
    鲁棒解析函数，处理 [source] 标签和断行。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 去除转义字符
        content = re.sub(r'\\', '', content)
        # 处理断行问题
        content = content.replace('\n', '|')
        # 清洗数据
        raw_items = [x.strip() for x in content.split('|') if x.strip()]
        
        data = []
        headers = {'Frame', 'IMU(ms)', 'YOLO(ms)', 'Flow(ms)', 'Cluster(ms)', 'Plot(ms)', 'Total(ms)'}
        
        current_row = []
        for item in raw_items:
            if item in headers or 'ms' in item or 'Frame' in item:
                continue
            try:
                val = float(item)
                current_row.append(val)
                if len(current_row) == 7:
                    data.append(current_row)
                    current_row = []
            except ValueError:
                continue
                
        df = pd.DataFrame(data, columns=['Frame', 'IMU', 'YOLO', 'Flow', 'Cluster', 'Plot', 'Total'])
        return df
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()

# --- 2. 加载与合并数据 ---
all_stages = []
stage_boundaries = []
current_global_frame = 0

for stage_name, file_path in files.items():
    df = parse_log_robust(file_path)
    if not df.empty:
        df['Global_Frame'] = range(current_global_frame + 1, current_global_frame + 1 + len(df))
        start_frame = current_global_frame
        end_frame = current_global_frame + len(df)
        stage_boundaries.append({
            'name': stage_name,
            'start': start_frame,
            'end': end_frame,
            'count': len(df)
        })
        current_global_frame = end_frame
        all_stages.append(df)

df_total = pd.concat(all_stages, ignore_index=True)

# --- 3. 绘图设置 ---
plt.figure(figsize=(16, 9))
ax = plt.gca()

x = df_total['Global_Frame']

# 【关键点 1】：区分数据列名和显示标签名
columns = ['IMU', 'YOLO', 'Flow', 'Cluster', 'Plot']
labels = ['Residual Field', 'Semantic(YOLO)', 'Kinematic-Aware', 'Dynamic Mask', 'Plot']
colors = ['#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#3498db'] 

# 提取数据
y_data = [df_total[col].clip(lower=0) for col in columns]

# 绘制堆叠图
ax.stackplot(x, y_data, labels=labels, colors=colors, alpha=0.85)

# 绘制总耗时曲线
ax.plot(x, df_total['Total'], color='black', linewidth=1, alpha=0.5, label='Total Latency')

# --- 4. 阶段背景与大标题 ---
stage_bg_colors = ['#e8f8f5', '#fef9e7', '#fdedec'] 

for i, stage in enumerate(stage_boundaries):
    # 背景着色
    ax.axvspan(stage['start'], stage['end'], color=stage_bg_colors[i], zorder=-10, alpha=0.6)
    
    # 虚线分割
    if i < len(stage_boundaries) - 1:
        ax.axvline(stage['end'], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 【关键点 2】：加大 Stage 名字字体 (fontsize=20)
    mid_point = (stage['start'] + stage['end']) / 2
    y_max = df_total['Total'].max()
    ax.text(mid_point, y_max * 1.08, stage['name'], 
            ha='center', va='bottom', fontsize=20, fontweight='bold', 
            color='#333333', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4.0))

# --- 5. 格式化微调 ---
# 设置 X 轴范围
ax.set_xlim(0, current_global_frame)

# 【关键点 3】：增加顶部留白 (1.3倍)，防止 Stage 文字被挡住
ax.set_ylim(0, df_total['Total'].max() * 1.35)

# 加大轴标签字体
ax.set_xlabel('Frames', fontsize=16, fontweight='bold')
ax.set_ylabel('Processing Latency (ms)', fontsize=16, fontweight='bold')

# 加大刻度数字字体
ax.tick_params(axis='both', labelsize=12)

ax.grid(axis='y', linestyle='--', alpha=0.3)

# 【关键点 4】：加大 Legend 字体 (fontsize=16) 并优化排版
ax.legend(loc='upper left', 
          ncol=3,           # 分 3 列显示更美观
          frameon=True, 
          fontsize=16,      # 字体加大
          edgecolor='gray',
          handletextpad=0.5,
          columnspacing=1.5)

plt.tight_layout()
plt.savefig('improved_latency_plot.png', dpi=300)
print("分析完成。图片已保存为 'improved_latency_plot.png'")