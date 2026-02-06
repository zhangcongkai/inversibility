import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# 0. 检查当前工作目录和文件
# =====================================================
print("=" * 60)
print("检查工作目录和文件...")
print("=" * 60)

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Python文件所在目录: {current_dir}")
print(f"当前工作目录: {os.getcwd()}")

# 尝试多个可能的模型文件路径
possible_paths = [
    "best_model.h5",  # 相对路径
    os.path.join(current_dir, "best_model.h5"),  # 绝对路径
    os.path.join(os.getcwd(), "best_model.h5")  # 基于工作目录的路径
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        print(f"✅ 找到模型文件: {path}")
        break

if model_path is None:
    print("❌ 错误: 在所有可能位置都找不到 best_model.h5 文件")
    print("当前目录下的文件:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    exit(1)

# =====================================================
# 1. GF(2) Rank Calculation
# =====================================================
def gf2_rank_bitpack(matrix):
    n = matrix.shape[0]
    rows = []

    for i in range(n):
        val = 0
        for j in range(n):
            if matrix[i, j]:
                val |= (1 << (n - 1 - j))
        rows.append(val)

    rank = 0
    for col in range(n):
        bit = 1 << (n - 1 - col)
        pivot = -1

        for r in range(rank, n):
            if rows[r] & bit:
                pivot = r
                break

        if pivot == -1:
            continue

        rows[rank], rows[pivot] = rows[pivot], rows[rank]

        for r in range(rank + 1, n):
            if rows[r] & bit:
                rows[r] ^= rows[rank]

        rank += 1

    return rank

# =====================================================
# 2. Generate 8×8 Test Data
# =====================================================
def generate_new_test_data(num_samples=10000):
    size = 8

    matrices = np.random.randint(
        0, 2, size=(num_samples, size, size)
    )

    labels = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        labels[i] = int(
            gf2_rank_bitpack(matrices[i]) == size
        )

    # CNN input format
    matrices = matrices.reshape(-1, size, size, 1).astype(np.float32)

    return matrices, labels

# =====================================================
# 3. Generate Test Dataset
# =====================================================
print("\n" + "=" * 60)
print("生成测试数据...")
print("=" * 60)

NUM_TEST = 10000
X_test, y_test = generate_new_test_data(NUM_TEST)

print(f"测试数据生成完成")
print(f"测试样本数: {NUM_TEST}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"可逆矩阵比例: {y_test.mean():.4f}")

# =====================================================
# 4. Load Trained Model
# =====================================================
print("\n" + "=" * 60)
print("加载模型...")
print("=" * 60)

try:
    print(f"正在加载模型: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("模型结构信息:")
    try:
        # 尝试查看文件信息
        file_size = os.path.getsize(model_path)
        print(f"模型文件大小: {file_size / 1024:.2f} KB")
    except:
        pass
    exit(1)

# =====================================================
# 5. Prediction
# =====================================================
print("\n" + "=" * 60)
print("进行预测...")
print("=" * 60)

y_prob = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

print(f"预测完成")
print(f"预测概率范围: [{y_prob.min():.6f}, {y_prob.max():.6f}]")

threshold = 0.5
y_pred = (y_prob.flatten() >= threshold).astype(int)

print(f"使用阈值: {threshold}")
print(f"预测为正类(可逆)的数量: {y_pred.sum()} ({y_pred.mean()*100:.2f}%)")

# =====================================================
# 6. Calculate Confusion Matrix
# =====================================================
print("\n" + "=" * 60)
print("计算评估指标...")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate performance metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 混淆矩阵:")
print(f"          预测负类  预测正类")
print(f"真实负类    {tn:6d}      {fp:6d}")
print(f"真实正类    {fn:6d}      {tp:6d}")

print("\n📈 性能指标:")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"F1分数 (F1-Score):  {f1:.4f}")

# =====================================================
# 7. Create Confusion Matrix Visualization
# =====================================================
print("\n" + "=" * 60)
print("创建可视化图表...")
print("=" * 60)

plt.figure(figsize=(10, 8))

# Main plot: Confusion matrix heatmap
plt.subplot(1, 2, 1)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-invertible", "Invertible"],
    yticklabels=["Non-invertible", "Invertible"],
    cbar_kws={'label': 'Count'}
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("8×8 Matrix Invertibility Confusion Matrix")

# Right side: Performance metrics
plt.subplot(1, 2, 2)
# Hide axes
plt.axis('off')

# Create table for metrics display
metrics_text = f"""Performance Metrics Summary:

Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1-Score:  {f1:.4f}

Confusion Matrix Values:
TN = {tn}
FP = {fp}
FN = {fn}
TP = {tp}

Test Dataset Statistics:
Total samples: {NUM_TEST}
Invertible: {y_test.sum()} ({y_test.mean()*100:.1f}%)
Non-invertible: {NUM_TEST - y_test.sum()} ({100 - y_test.mean()*100:.1f}%)

Model: {os.path.basename(model_path)}"""

plt.text(0.1, 0.5, metrics_text, 
         fontsize=11, 
         fontfamily='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f"8×8 Binary Matrix Invertibility Detection", fontsize=16, fontweight='bold')
plt.tight_layout()

# Save image
output_filename = 'confusion_matrix_8x8.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"✅ 混淆矩阵图片已保存为: {output_filename}")

# Display image
plt.show()

# =====================================================
# 8. Save Confusion Matrix Data
# =====================================================
np.save("confusion_matrix_8x8.npy", cm)

# Save summary results to text file
with open("results_summary.txt", "w") as f:
    f.write("8×8 Matrix Invertibility Detection Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model used: {model_path}\n")
    f.write(f"Test samples: {NUM_TEST}\n")
    f.write(f"Invertible matrix ratio: {y_test.mean():.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall:    {rec:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")

print("✅ 结果摘要已保存为: results_summary.txt")

print("\n" + "=" * 60)
print("✅ 分析完成!")
print("=" * 60)