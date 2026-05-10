import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('output/visualizations/comparison/comparison_methods_means.csv')

# 过滤掉Human行
df = df[df['method'] != 'Human']

# 重命名AMPS-GPT5.1为Ours
df['method'] = df['method'].replace('AMPS-GPT5.1', 'Ours')

# 创建图表
plt.figure(figsize=(10, 6))
bars = plt.bar(df['method'], df['overall_f1_score'], color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

# 在每个柱子上添加具体数值
for i, (bar, score) in enumerate(zip(bars, df['overall_f1_score'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置图表样式
plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
plt.ylim(0, max(df['overall_f1_score']) * 1.15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存图表
plt.savefig('output/visualizations/comparison/f1_score_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存到: output/visualizations/comparison/f1_score_comparison.png")
