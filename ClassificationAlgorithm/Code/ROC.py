import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 1. 设置CSV文件路径 ----------------------------------------------------
csv_file_path = r"E:\Study\Hornbill\Time-Series-Library-main\results\classification_EthanolConcentration_Pyraformer_UEA_ftM_sl250_ll48_pl0_dm8_nh16_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_'Exp'_0\.csv"

# 检查文件是否存在
if not os.path.exists(csv_file_path):
    # 尝试去掉末尾的反斜杠
    csv_file_path = r"E:\Study\Hornbill\Time-Series-Library-main\results\classification_EthanolConcentration_Pyraformer_UEA_ftM_sl250_ll48_pl0_dm8_nh16_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_'Exp'_0.csv"

if not os.path.exists(csv_file_path):
    print(f"错误: 找不到文件")
    print(f"尝试的路径1: {csv_file_path}")
    exit()

print("正在读取CSV文件...")
print(f"文件路径: {csv_file_path}")

# 2. 读取CSV文件 ----------------------------------------------------
try:
    # 尝试不同编码读取
    try:
        data = pd.read_csv(csv_file_path)
    except UnicodeDecodeError:
        data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

    print("✓ 数据读取成功!")
    print(f"✓ 数据形状: {data.shape} (行数: {data.shape[0]}, 列数: {data.shape[1]})")

    # 显示前几行数据
    print("\n数据预览（前5行）:")
    print(data.head())

    # 显示列名
    print(f"\n所有列名 ({len(data.columns)} 列):")
    for i, col in enumerate(data.columns):
        print(f"  {i:2d}. {col}")

except Exception as e:
    print(f"✗ 读取CSV文件时出错: {e}")
    exit()

# 3. 分析数据结构 ----------------------------------------------------
print("\n" + "=" * 60)
print("数据结构分析")
print("=" * 60)

# 查看数据的基本信息
print("数据基本信息:")
print(data.info())

# 检查数据列类型
print("\n前3行的数据:")
for i in range(min(3, len(data))):
    print(f"行 {i}: {data.iloc[i].tolist()[:5]}...")  # 只显示前5个值

# 4. 识别真实标签和预测概率 ----------------------------------------------------
# 寻找可能的标签列
label_candidates = ['true', 'label', 'y_true', 'target', 'actual', 'ground_truth']
pred_candidates = ['pred', 'prob', 'score', 'prediction', 'y_pred']

label_column = None
prob_columns = []

# 先尝试找到标签列
for col in data.columns:
    col_lower = col.lower()
    for candidate in label_candidates:
        if candidate in col_lower:
            label_column = col
            print(f"✓ 找到标签列: '{label_column}'")
            break
    if label_column:
        break

# 如果没找到标准标签列，尝试第一列
if label_column is None:
    print("⚠ 未找到标准标签列，尝试使用第一列")
    label_column = data.columns[0]
    print(f"使用第一列作为标签: '{label_column}'")

# 提取真实标签
true_labels = data[label_column].values
print(f"✓ 提取真实标签，形状: {true_labels.shape}")
print(f"  标签值示例: {true_labels[:10]}")
print(f"  唯一标签值: {np.unique(true_labels)}")
print(f"  标签数据类型: {true_labels.dtype}")

# 寻找概率列
# 首先排除标签列
remaining_columns = [col for col in data.columns if col != label_column]

# 检查剩余列是否是概率（值在0-1之间）
for col in remaining_columns:
    sample_values = data[col].values[:5]
    # 检查是否是数值型
    if pd.api.types.is_numeric_dtype(data[col]):
        # 检查值范围
        if data[col].min() >= 0 and data[col].max() <= 1:
            prob_columns.append(col)
        else:
            print(f"⚠ 列 '{col}' 的值不在0-1范围内: [{data[col].min():.3f}, {data[col].max():.3f}]")

print(f"\n✓ 找到 {len(prob_columns)} 个概率列")
if len(prob_columns) > 0:
    print(f"概率列: {prob_columns[:5]}...")  # 只显示前5个

    # 提取概率矩阵
    pred_probs = data[prob_columns].values
    n_classes = pred_probs.shape[1]
    print(f"✓ 概率矩阵形状: {pred_probs.shape}")

    # 检查概率和
    print("\n前3个样本的概率和:")
    for i in range(min(3, len(true_labels))):
        prob_sum = pred_probs[i].sum()
        print(f"  样本 {i} (标签={true_labels[i]}): 概率和={prob_sum:.4f}")

else:
    # 如果没有找到概率列，尝试将其他数值列视为预测分数
    print("⚠ 未找到标准概率列，尝试使用其他数值列")

    # 选择数值列（排除标签列）
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if label_column in numeric_columns:
        numeric_columns.remove(label_column)

    if len(numeric_columns) > 0:
        print(f"使用数值列作为预测分数: {numeric_columns}")
        pred_probs = data[numeric_columns].values
        n_classes = pred_probs.shape[1]
        print(f"预测分数矩阵形状: {pred_probs.shape}")

        # 可能需要归一化
        print("对预测分数进行归一化...")
        row_sums = pred_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        pred_probs = pred_probs / row_sums

        prob_columns = numeric_columns
    else:
        print("✗ 错误: 没有找到可用的数值列")
        exit()

# 5. 数据预处理 ----------------------------------------------------
print("\n" + "=" * 60)
print("数据预处理")
print("=" * 60)

# 确保标签是整数
if true_labels.dtype != np.int64 and true_labels.dtype != np.int32:
    print(f"⚠ 标签数据类型为 {true_labels.dtype}，尝试转换为整数...")
    try:
        # 尝试直接转换
        true_labels = true_labels.astype(int)
        print("✓ 标签转换为整数成功")
    except:
        # 如果失败，尝试映射
        print("尝试使用标签映射...")
        unique_labels = np.unique(true_labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        true_labels = np.array([label_mapping[label] for label in true_labels])
        print(f"标签映射: {label_mapping}")

# 检查标签范围
unique_labels = np.unique(true_labels)
n_unique_labels = len(unique_labels)
print(f"✓ 唯一标签数: {n_unique_labels}")
print(f"✓ 标签范围: {unique_labels.min()} 到 {unique_labels.max()}")

# 如果标签不是从0开始，重新映射
if unique_labels.min() != 0:
    print("⚠ 标签不是从0开始，重新映射...")
    true_labels = true_labels - unique_labels.min()
    unique_labels = np.unique(true_labels)
    print(f"重新映射后的标签范围: {unique_labels.min()} 到 {unique_labels.max()}")

# 6. 计算综合ROC曲线（所有类别合并为一条） ----------------------------------------------------
print("\n" + "=" * 60)
print("计算综合ROC曲线")
print("=" * 60)

print("使用微平均方法计算综合ROC曲线...")

# 方法1: 微平均（将所有样本的所有类别展平）
y_true_all = []
y_score_all = []

for i in range(len(true_labels)):
    # 创建one-hot编码的真实标签
    true_one_hot = np.zeros(n_classes)
    true_one_hot[true_labels[i]] = 1

    y_true_all.extend(true_one_hot)
    y_score_all.extend(pred_probs[i])

y_true_all = np.array(y_true_all)
y_score_all = np.array(y_score_all)

print(f"  展平后的数据形状:")
print(f"    y_true_all: {y_true_all.shape}")
print(f"    y_score_all: {y_score_all.shape}")

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)

print(f"  ROC曲线点数: {len(fpr)}")
print(f"  综合AUC: {roc_auc:.4f}")

# 7. 创建可视化图形 ----------------------------------------------------
print("\n" + "=" * 60)
print("创建ROC曲线图")
print("=" * 60)

plt.figure(figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 绘制主ROC曲线 - 加粗到5.0
plt.plot(fpr, tpr, color='#2E86AB', lw=6.0,  # 增加线宽到5.0
         label=f'Combined ROC (AUC = {roc_auc:.3f})',
         alpha=0.9, zorder=5)

# 绘制对角线（随机猜测线）
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=5.0,  # 增加线宽
         alpha=0.7, label='Random Classifier (AUC = 0.500)', zorder=1)

# 设置图形属性
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)',
           fontsize=27, fontweight='bold', labelpad=12)  # 增大字体
plt.ylabel('True Positive Rate (Sensitivity)',
           fontsize=27, fontweight='bold', labelpad=12)  # 增大字体

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--', zorder=0)

# 8. 计算置信区间（使用bootstrap） ----------------------------------------------------
print("计算AUC置信区间...")
n_bootstrap = 1000
bootstrap_aucs = []

np.random.seed(42)
n_samples = len(true_labels)

for i in range(n_bootstrap):
    # 有放回抽样
    indices = np.random.choice(n_samples, n_samples, replace=True)

    # 使用相同的索引获取样本
    sample_true = true_labels[indices]
    sample_probs = pred_probs[indices]

    # 转换为二分类格式
    y_true_sample = []
    y_score_sample = []

    for j in range(len(sample_true)):
        true_one_hot = np.zeros(n_classes)
        true_one_hot[sample_true[j]] = 1
        y_true_sample.extend(true_one_hot)
        y_score_sample.extend(sample_probs[j])

    y_true_sample = np.array(y_true_sample)
    y_score_sample = np.array(y_score_sample)

    # 计算AUC
    try:
        sample_auc = roc_auc_score(y_true_sample, y_score_sample)
        bootstrap_aucs.append(sample_auc)
    except:
        continue

# 计算置信区间
if bootstrap_aucs:
    bootstrap_aucs = np.array(bootstrap_aucs)
    auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
    auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
    auc_mean = np.mean(bootstrap_aucs)
    auc_std = np.std(bootstrap_aucs)

    print(f"  Bootstrap AUC统计:")
    print(f"    均值: {auc_mean:.4f}")
    print(f"    标准差: {auc_std:.4f}")
    print(f"    95%置信区间: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
else:
    auc_ci_lower = auc_ci_upper = roc_auc
    auc_mean = roc_auc
    auc_std = 0
    print("  ⚠ 无法计算置信区间")

# 9. 添加填充和标记最优阈值点 ----------------------------------------------------
# 在ROC曲线下添加浅色填充
plt.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB', zorder=4)

# 找到Youden指数最大的点（最优阈值）
youden_idx = np.argmax(tpr - fpr)
optimal_fpr = fpr[youden_idx]
optimal_tpr = tpr[youden_idx]
optimal_threshold = thresholds[youden_idx] if youden_idx < len(thresholds) else 0.5

# 标记最优点
plt.scatter(optimal_fpr, optimal_tpr, color='#D1495B', s=200,  # 增大点的大小
            zorder=16, edgecolors='black', linewidth=3.0,  # 增加边框宽度
            label=f'Optimal Threshold ({optimal_threshold:.3f})',
            marker='*')

# 10. 添加统计信息框 ----------------------------------------------------
# 确定性能评价
if roc_auc >= 0.9:
    performance = "Excellent"
    perf_color = "#2E7D32"  # 深绿
elif roc_auc >= 0.8:
    performance = "Good"
    perf_color = "#388E3C"  # 绿
elif roc_auc >= 0.7:
    performance = "Fair"
    perf_color = "#F57C00"  # 橙
elif roc_auc >= 0.6:
    performance = "Poor"
    perf_color = "#D32F2F"  # 红
else:
    performance = "Fail"
    perf_color = "#C2185B"  # 深红

# ===== 修改：使用指定的真/假阳性率值 =====
# 直接使用您指定的值
final_tpr = 0.751  # 真阳性率
final_fpr = 0.247  # 假阳性率
final_tnr = 1 - final_fpr  # 真阴性率
final_fnr = 1 - final_tpr  # 假阴性率
youden_index = final_tpr - final_fpr  # Youden指数
balanced_acc = (final_tpr + final_tnr) / 2  # 平衡准确率

# ===== 左上角统计信息框（黄色） =====
# 创建左上角统计信息文本
stats_text_left = f'Dataset Summary:\n'
stats_text_left += f'• Samples: {len(true_labels):,}\n'
stats_text_left += f'• Classes: {n_unique_labels}\n'
stats_text_left += f'• Features: {n_classes}\n'
stats_text_left += f'• AUC: {roc_auc:.3f}\n'
stats_text_left += f'• 95% CI: [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]\n'
stats_text_left += f'• Performance: {performance}'

# 将统计信息框放在左上角（黄色）
plt.text(0.02, 0.98, stats_text_left, transform=plt.gca().transAxes,
         fontsize=19, verticalalignment='top', fontweight='bold',
         horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4',
                   alpha=0.95, edgecolor='#FFB300', linewidth=3),
         color=perf_color, zorder=20)

# 11. 添加图例（先添加图例，再添加右下角统计框） ----------------------------------------------------
# 设置图例 - 右下角
legend = plt.legend(loc="lower right", fontsize=22, frameon=True,
           facecolor='white', framealpha=0.95, edgecolor='black',
           borderpad=1.5, labelspacing=1.5, handlelength=2.5,
           prop={'weight': 'bold', 'size': 16})

# ===== 右下角统计信息框（绿色）- 放在图例上方，保证在大框内 =====
# 创建右下角统计信息文本（真假阳性率相关参数）
stats_text_right = f'Performance Metrics:\n'
stats_text_right += f'• TPR (Sensitivity): {final_tpr:.3f}\n'
stats_text_right += f'• FPR (1-Specificity): {final_fpr:.3f}\n'
stats_text_right += f'• TNR (Specificity): {final_tnr:.3f}\n'
stats_text_right += f'• FNR (Miss Rate): {final_fnr:.3f}\n'
stats_text_right += f'• Youden Index: {youden_index:.3f}\n'
stats_text_right += f'• Balanced Acc: {balanced_acc:.3f}'

# 获取图例的位置信息（在axes坐标系中）
# 使用更保守的方法计算绿色框位置，确保在大框内
ax = plt.gca()

# 计算绿色框的合适位置（在axes坐标系中）
# 将绿色框放在图例上方，但在大框内部
green_box_x = 0.98  # 靠近右边，但留出空间
green_box_y = 0.25  # 适当高度，确保在大框内且不重叠

# 调整图例位置，为绿色框留出空间
legend.set_bbox_to_anchor((1, 0.00))  # 将图例向下移动

# 将绿色统计信息框放在图例上方（在axes坐标系中）
plt.text(green_box_x, green_box_y, stats_text_right, transform=ax.transAxes,
         fontsize=17, verticalalignment='bottom', fontweight='bold',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9',
                   alpha=0.95, edgecolor='#4CAF50', linewidth=3),
         color='#2E7D32', zorder=30)  # zorder设为30确保在最上层

# 12. 样式调整 ----------------------------------------------------
# 调整刻度 - 加粗变大
plt.xticks(fontsize=30, fontweight='bold')  # 增大刻度字体并加粗
plt.yticks(fontsize=30, fontweight='bold')  # 增大刻度字体并加粗

# 设置刻度线
ax.tick_params(axis='both', which='major', width=3, length=8)  # 增大刻度线
ax.tick_params(axis='both', which='minor', width=3, length=5)

# ===== 修改：设置边框为黑色，所有边框加粗 =====
# 设置所有边框为黑色并加粗
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(5)  # 所有边框都加粗

# 13. 保存和显示图形 ----------------------------------------------------
plt.tight_layout()

# 保存图形
base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
if base_name.endswith('_'):
    base_name = base_name[:-1]

output_dir = r"E:\Study\Hornbill\Time-Series-Library-main\results\roc_plots"
os.makedirs(output_dir, exist_ok=True)

# 保存为PNG
png_output = os.path.join(output_dir, f'{base_name}_combined_roc.png')
plt.savefig(png_output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ ROC曲线图已保存为: {png_output}")

# 保存为PDF
pdf_output = os.path.join(output_dir, f'{base_name}_combined_roc.pdf')
plt.savefig(pdf_output, bbox_inches='tight', facecolor='white', format='pdf')
print(f"✓ ROC曲线图已保存为: {pdf_output}")



# 14. 打印详细报告 ----------------------------------------------------
print("\n" + "=" * 80)
print("综合ROC曲线分析报告")
print("=" * 80)

print(f"\n📊 数据集概览:")
print(f"  文件: {os.path.basename(csv_file_path)}")
print(f"  总样本数: {len(true_labels):,}")
print(f"  唯一标签数: {n_unique_labels}")
print(f"  预测特征数: {n_classes}")

print(f"\n📈 ROC曲线分析结果:")
print(f"  综合AUC分数: {roc_auc:.4f}")
print(f"  AUC平均值 (bootstrap): {auc_mean:.4f}")
print(f"  AUC标准差: {auc_std:.4f}")
print(f"  95%置信区间: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]")
print(f"  性能评价: {performance}")

print(f"\n📊 指定的性能指标:")
print(f"  真阳性率 (TPR): {final_tpr:.3f}")
print(f"  假阳性率 (FPR): {final_fpr:.3f}")
print(f"  真阴性率 (TNR): {final_tnr:.3f}")
print(f"  假阴性率 (FNR): {final_fnr:.3f}")
print(f"  Youden指数: {youden_index:.3f}")
print(f"  平衡准确率: {balanced_acc:.3f}")

# ===== 新增：详细的TPR和FPR输出 =====
print(f"\n📊 ROC曲线关键点详细信息:")
print("-" * 70)
print(f"{'Threshold':<12} {'FPR (假阳性率)':<18} {'TPR (真阳性率)':<18} {'Youden指数':<12} {'性能状态'}")
print("-" * 70)

# 定义关键阈值点
key_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for thr in key_thresholds:
    # 找到最接近的阈值点
    thr_idx = np.argmin(np.abs(thresholds - thr))
    if thr_idx < len(fpr):
        current_fpr = fpr[thr_idx]
        current_tpr = tpr[thr_idx]
        youden = current_tpr - current_fpr

        # 判断性能状态
        if thr == optimal_threshold:
            status = "★ 最优阈值"
        elif youden > 0.7:
            status = "优秀"
        elif youden > 0.5:
            status = "良好"
        elif youden > 0.3:
            status = "一般"
        else:
            status = "较差"

        print(f"{thr:<12.3f} {current_fpr:<18.4f} {current_tpr:<18.4f} {youden:<12.4f} {status}")

# 添加最优阈值点的详细信息
print("-" * 70)
print(
    f"{'最优阈值':<12} {optimal_threshold:<18.4f} {optimal_tpr:<18.4f} {optimal_tpr - optimal_fpr:<12.4f} {'★ 最佳性能'}")
print("-" * 70)

print(f"\n🔍 关键性能指标解释:")
print("  • 真阳性率 (TPR/Recall/Sensitivity):")
print(f"    - 表示实际为正类的样本中被正确预测的比例: {final_tpr:.1%}")
print(f"    - 这意味着模型能正确识别 {final_tpr * 100:.1f}% 的正样本")

print(f"\n  • 假阳性率 (FPR/1-Specificity):")
print(f"    - 表示实际为负类的样本中被错误预测为正类的比例: {final_fpr:.1%}")
print(f"    - 这意味着模型将 {final_fpr * 100:.1f}% 的负样本误判为正样本")

print(f"\n  • 真阴性率 (TNR/Specificity):")
print(f"    - 表示实际为负类的样本中被正确预测的比例: {final_tnr:.1%}")

print(f"\n  • 假阴性率 (FNR/Miss Rate):")
print(f"    - 表示实际为正类的样本中被错误预测为负类的比例: {final_fnr:.1%}")

print(f"\n📊 性能指标汇总表:")
print("-" * 85)
print(f"{'指标':<25} {'数值':<15} {'百分比':<15} {'解释':<30}")
print("-" * 85)
print(f"{'真阳性率 (TPR)':<25} {final_tpr:<15.4f} {final_tpr * 100:<15.1f}% {'敏感性/召回率':<30}")
print(f"{'假阳性率 (FPR)':<25} {final_fpr:<15.4f} {final_fpr * 100:<15.1f}% {'1 - 特异度':<30}")
print(f"{'真阴性率 (TNR)':<25} {final_tnr:<15.4f} {final_tnr * 100:<15.1f}% {'特异度':<30}")
print(f"{'假阴性率 (FNR)':<25} {final_fnr:<15.4f} {final_fnr * 100:<15.1f}% {'漏诊率':<30}")
print(f"{'Youden指数':<25} {youden_index:<15.4f} {'':<15} {'TPR - FPR (最大为1)':<30}")
print(f"{'平衡准确率':<25} {balanced_acc:<15.4f} {'':<15} {'(TPR + TNR)/2':<30}")
print("-" * 85)

# 15. 保存详细结果 ----------------------------------------------------
print(f"\n💾 保存详细结果...")

# 保存ROC曲线数据点（包括TPR和FPR）
roc_data = pd.DataFrame({
    'Threshold': thresholds[:len(fpr)],
    'FPR': fpr,
    'TPR': tpr,
    'Specificity': 1 - fpr,
    'Youden_Index': tpr - fpr,
    'F1_Score': 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-10)  # 避免除以零
})

roc_data_output = os.path.join(output_dir, f'{base_name}_roc_data_detailed.csv')
roc_data.to_csv(roc_data_output, index=False)
print(f"✓ ROC曲线详细数据已保存到: {roc_data_output}")

# 保存综合统计
summary_stats = {
    'File': os.path.basename(csv_file_path),
    'Total_Samples': len(true_labels),
    'Unique_Classes': n_unique_labels,
    'Features': n_classes,
    'Combined_AUC': roc_auc,
    'AUC_Mean_Bootstrap': auc_mean,
    'AUC_Std_Bootstrap': auc_std,
    'AUC_CI_Lower': auc_ci_lower,
    'AUC_CI_Upper': auc_ci_upper,
    'Optimal_Threshold': optimal_threshold,
    'Optimal_FPR': optimal_fpr,
    'Optimal_TPR': optimal_tpr,
    'Specified_TPR': final_tpr,
    'Specified_FPR': final_fpr,
    'Specified_TNR': final_tnr,
    'Specified_FNR': final_fnr,
    'Youden_Index': youden_index,
    'Balanced_Accuracy': balanced_acc,
    'Performance_Level': performance,
    'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_df = pd.DataFrame([summary_stats])
summary_output = os.path.join(output_dir, f'{base_name}_roc_summary.csv')
summary_df.to_csv(summary_output, index=False)
print(f"✓ 综合统计已保存到: {summary_output}")

print("\n" + "=" * 80)
print("✅ ROC曲线分析完成!")
print(f"所有结果已保存到: {output_dir}")
print("=" * 80)


