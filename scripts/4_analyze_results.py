import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# --- 路径适配 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config

# --- 全局风格设置 ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
# 定义一致的配色方案
MODEL_PALETTE = {"DenseNet": "#4e79a7", "Swin": "#e15759"}  # 蓝/红
METHOD_PALETTE = {"Grad-CAM": "#76b7b2", "IG": "#f28e2b"}  # 青/橙


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def enrich_data_with_lesion_size(df):
    """
    [核心修复] 如果评估结果中缺少 lesion_size，从原始标签文件中计算并补全。
    无需重新运行评估脚本。
    """
    if 'lesion_size' in df.columns:
        return df

    print("🔧 检测到缺少 'lesion_size' 列，正在从原始标签文件补全...")
    label_path = Config.RAW_LABEL_CSV
    if not os.path.exists(label_path):
        print(f"⚠️ 无法找到原始标签文件: {label_path}，将跳过病灶大小分析。")
        return df

    try:
        # 读取原始标签
        labels = pd.read_csv(label_path)
        # 计算每个病人的病灶总面积 (RSNA原图尺寸为 1024x1024)
        # 过滤掉 NaN (正常样本)
        labels = labels.dropna(subset=['x', 'y', 'width', 'height'])
        labels['area'] = labels['width'] * labels['height']

        # 按病人聚合（一个病人可能有多个框）
        patient_areas = labels.groupby('patientId')['area'].sum().reset_index()

        # 计算占比 (Area / 1024^2)
        patient_areas['lesion_size_calculated'] = patient_areas['area'] / (1024 * 1024)

        # 合并到主数据
        df = df.merge(patient_areas[['patientId', 'lesion_size_calculated']], on='patientId', how='left')
        # 填充 NaN (没有框的即为 0)
        df['lesion_size_calculated'] = df['lesion_size_calculated'].fillna(0)
        # 重命名
        df['lesion_size'] = df['lesion_size_calculated']
        print(f"✅ 已成功补全 {len(df)} 条数据的病灶大小信息。")

    except Exception as e:
        print(f"⚠️ 补全数据失败: {e}")

    return df


def load_data():
    """加载并合并评估结果数据"""
    df_list = []
    for model in ['densenet121', 'swin_t']:
        path = os.path.join(Config.OUTPUT_DIR, 'results', f'audit_{model}.csv')
        if os.path.exists(path):
            try:
                d = pd.read_csv(path)
                if len(d) > 0:
                    d['Architecture'] = 'DenseNet' if 'densenet' in model else 'Swin'
                    df_list.append(d)
            except Exception as e:
                print(f"❌ 读取错误 {path}: {e}")

    if not df_list:
        print("❌ 未找到评估结果，请先运行 3_run_evaluation.py")
        return None

    full_df = pd.concat(df_list, ignore_index=True)

    # 尝试补全缺失的病灶大小数据
    full_df = enrich_data_with_lesion_size(full_df)

    return full_df


def save_plot(filename):
    """辅助函数：保存图片"""
    save_path = os.path.join(Config.OUTPUT_DIR, 'figures_final', filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 已保存: {filename}")


# ==================================================================================
# Plot 1: IoU Distribution
# ==================================================================================
def plot_iou_distribution(df):
    print("🎨 Plot 1: IoU Distribution...")
    plt.figure(figsize=(10, 6))

    iou_cols = [c for c in ['iou_gc', 'iou_ig'] if c in df.columns]
    if not iou_cols: return

    df_melt = df.melt(id_vars=['Architecture'], value_vars=iou_cols, var_name='Method', value_name='IoU')
    df_melt['Method'] = df_melt['Method'].map({'iou_gc': 'Grad-CAM', 'iou_ig': 'IG'})

    sns.boxplot(data=df_melt, x='Architecture', y='IoU', hue='Method',
                palette=METHOD_PALETTE, showfliers=False, width=0.5, linewidth=1.5)

    sns.stripplot(data=df_melt, x='Architecture', y='IoU', hue='Method',
                  dodge=True, alpha=0.3, color='.2', size=3)

    plt.title("Clinical Alignment: IoU Distribution", fontweight='bold', pad=15)
    plt.ylabel("IoU Score (Higher is Better)")
    plt.xlabel("")
    plt.legend(title='XAI Method', loc='upper right')
    save_plot('1_IoU_Distribution.png')


# ==================================================================================
# Plot 2: Hit Rate Comparison
# ==================================================================================
def plot_hit_rate(df):
    print("🎨 Plot 2: Hit Rate...")
    plt.figure(figsize=(8, 6))

    hit_cols = [c for c in ['hit_gc', 'hit_ig'] if c in df.columns]
    if not hit_cols: return

    df_agg = df.groupby(['Architecture'])[hit_cols].mean().reset_index()
    df_melt = df_agg.melt(id_vars=['Architecture'], var_name='Method', value_name='Hit Rate')
    df_melt['Method'] = df_melt['Method'].map({'hit_gc': 'Grad-CAM', 'hit_ig': 'IG'})

    ax = sns.barplot(data=df_melt, x='Architecture', y='Hit Rate', hue='Method',
                     palette=METHOD_PALETTE, alpha=0.9, edgecolor=".2")

    plt.ylim(0, 1.1)
    plt.title("Pointing Game (Hit Rate)", fontweight='bold', pad=15)
    plt.ylabel("Hit Rate (Precision)")
    plt.xlabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=12)

    save_plot('2_Hit_Rate.png')


# ==================================================================================
# Plot 3: Faithfulness vs Alignment
# ==================================================================================
def plot_faithfulness_alignment(df):
    print("🎨 Plot 3: Faithfulness vs Alignment...")
    plt.figure(figsize=(10, 7))

    if 'fidelity_gc' not in df.columns: return

    plot_df = df[df['fidelity_gc'] < df['fidelity_gc'].quantile(0.98)]

    sns.scatterplot(data=plot_df, x='fidelity_gc', y='iou_gc', hue='Architecture', style='Architecture',
                    palette=MODEL_PALETTE, s=80, alpha=0.6, edgecolor='w')

    for arch in ['DenseNet', 'Swin']:
        if arch in df['Architecture'].unique():
            subset = plot_df[plot_df['Architecture'] == arch]
            if len(subset) > 1:
                sns.regplot(data=subset, x='fidelity_gc', y='iou_gc', scatter=False,
                            color=MODEL_PALETTE[arch], line_kws={'linestyle': '--'})

    plt.title("Trade-off: Faithfulness vs. Alignment", fontweight='bold', pad=15)
    plt.xlabel("Deletion AUC (Lower = More Faithful)")
    plt.ylabel("IoU Score (Higher = Better Alignment)")
    plt.grid(True, alpha=0.3)
    save_plot('3_Faithfulness_vs_Alignment.png')


# ==================================================================================
# Plot 4: Uncertainty Distribution
# ==================================================================================
def plot_uncertainty_dist(df):
    print("🎨 Plot 4: Uncertainty Distribution...")
    plt.figure(figsize=(8, 6))

    if 'uncertainty_gc' not in df.columns: return

    plot_df = df[df['uncertainty_gc'] > 1e-5]
    if len(plot_df) < 5: return

    sns.violinplot(data=plot_df, x='Architecture', y='uncertainty_gc',
                   palette=MODEL_PALETTE, inner="quart", cut=0)

    plt.title("Model Uncertainty Distribution", fontweight='bold', pad=15)
    plt.ylabel("Uncertainty (Std Dev)")
    plt.xlabel("")
    save_plot('4_Uncertainty_Distribution.png')


# ==================================================================================
# Plot 5: Uncertainty vs Performance
# ==================================================================================
def plot_uncertainty_vs_iou(df):
    print("🎨 Plot 5: Uncertainty vs IoU...")
    plt.figure(figsize=(10, 7))

    if 'uncertainty_gc' not in df.columns: return

    plot_df = df[df['uncertainty_gc'] > 1e-5]

    sns.scatterplot(data=plot_df, x='uncertainty_gc', y='iou_gc', hue='Architecture',
                    palette=MODEL_PALETTE, alpha=0.5, s=60)

    sns.regplot(data=plot_df, x='uncertainty_gc', y='iou_gc', scatter=False,
                color=".2", line_kws={'linestyle': '--', 'label': 'Global Trend'})

    plt.title("Does Uncertainty Predict Failure?", fontweight='bold', pad=15)
    plt.xlabel("Model Uncertainty (Std Dev)")
    plt.ylabel("IoU Score")
    plt.legend()

    plt.text(0.05, 0.05, "Negative Slope = \nUncertainty Flags Errors",
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    save_plot('5_Uncertainty_vs_Performance.png')


# ==================================================================================
# Plot 6: Lesion Size Impact (修复版)
# ==================================================================================
def plot_lesion_size_impact(df):
    print("🎨 Plot 6: Lesion Size Impact...")

    # 检查关键列是否存在
    if 'lesion_size' not in df.columns:
        print("⚠️ 警告: 'lesion_size' 数据仍然缺失，无法生成图6。")
        return

    plt.figure(figsize=(10, 7))

    # 过滤掉 0 大小（可能是计算错误或无病灶）
    plot_df = df[df['lesion_size'] > 0]

    if len(plot_df) < 5:
        print("⚠️ 有效病灶数据不足，跳过图6。")
        return

    for arch in ['DenseNet', 'Swin']:
        if arch in df['Architecture'].unique():
            subset = plot_df[plot_df['Architecture'] == arch]
            sns.regplot(data=subset, x='lesion_size', y='iou_gc',
                        scatter_kws={'alpha': 0.2, 's': 30, 'color': MODEL_PALETTE[arch]},
                        line_kws={'color': MODEL_PALETTE[arch], 'label': f'{arch} Trend', 'linewidth': 3})

    plt.title("Impact of Lesion Size on Accuracy", fontweight='bold', pad=15)
    plt.xlabel("Relative Lesion Size (Area Ratio)")
    plt.ylabel("IoU Score")
    plt.ylim(0, 1.0)
    plt.legend()
    save_plot('6_Lesion_Size_Impact.png')


# ==================================================================================
# Plot 7: Method Consistency
# ==================================================================================
def plot_consistency(df):
    print("🎨 Plot 7: Method Consistency...")
    if 'iou_gc' not in df.columns or 'iou_ig' not in df.columns: return

    g = sns.JointGrid(data=df, x="iou_gc", y="iou_ig", hue="Architecture", palette=MODEL_PALETTE, height=8)
    g.plot_joint(sns.scatterplot, s=50, alpha=0.5, edgecolor="w")
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)

    g.ax_joint.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.5)

    g.fig.suptitle("Method Consistency: Grad-CAM vs IG", fontsize=16, fontweight='bold', y=1.02)
    g.set_axis_labels("Grad-CAM IoU", "Integrated Gradients IoU")

    save_path = os.path.join(Config.OUTPUT_DIR, 'figures_final', '7_Method_Consistency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: 7_Method_Consistency.png")


# ==================================================================================
# Plot 8: Sample-wise Parallel Coordinates (修复Pandas Warning)
# ==================================================================================
def plot_parallel_coordinates(df):
    print("🎨 Plot 8: Sample-wise Analysis...")
    plt.figure(figsize=(12, 6))

    metrics = ['iou_gc', 'hit_gc', 'fidelity_gc', 'uncertainty_gc']
    valid_metrics = [m for m in metrics if m in df.columns]
    if len(valid_metrics) < 3: return

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # [修复Warning] 使用 groupby.sample 代替 apply+lambda
    try:
        # 尝试对每个架构随机采样30个点
        subset = df.groupby('Architecture', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 30)),
                                                                    include_groups=True)
    except:
        # 兼容旧版pandas或报错回退
        subset = df.sample(n=min(len(df), 60))

    if len(subset) == 0: return

    norm_data = scaler.fit_transform(subset[valid_metrics])
    plot_data = pd.DataFrame(norm_data, columns=valid_metrics)
    plot_data['Architecture'] = subset['Architecture'].values

    pd.plotting.parallel_coordinates(plot_data, 'Architecture',
                                     color=[MODEL_PALETTE.get('DenseNet', 'b'), MODEL_PALETTE.get('Swin', 'r')],
                                     alpha=0.4)

    plt.title("Multi-Metric Sample Profiles (Normalized)", fontweight='bold', pad=15)
    plt.ylabel("Normalized Score")
    plt.xlabel("Metrics")
    plt.grid(alpha=0.3)

    save_plot('8_Sample_Parallel_Coords.png')


# ==================================================================================
# Main Execution
# ==================================================================================
def analyze():
    print("📊 Generating 8 Final Publication-Ready Plots...")
    ensure_dir(os.path.join(Config.OUTPUT_DIR, 'figures_final'))

    df = load_data()
    if df is None: return

    print(f"✅ Data Loaded & Enriched: {len(df)} samples.")

    plot_iou_distribution(df)
    plot_hit_rate(df)
    plot_faithfulness_alignment(df)
    plot_uncertainty_dist(df)
    plot_uncertainty_vs_iou(df)
    plot_lesion_size_impact(df)
    plot_consistency(df)
    plot_parallel_coordinates(df)

    print(f"\n🎉 All 8 plots saved to: {os.path.join(Config.OUTPUT_DIR, 'figures_final')}")


if __name__ == "__main__":
    analyze()