# CompareANNandKNNRegression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data  # 特征数据
y = mnist.target  # 目标标签

# 将目标标签转换为数值类型
y = y.astype(int)

# 创建回归目标：每个数字的平均像素值
y_reg = X.mean(axis=1)

# 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

# 训练人工神经网络回归模型
ann_regressor = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
ann_regressor.fit(X_train, y_train)

# 训练k近邻回归模型
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_ann = ann_regressor.predict(X_test)
y_pred_knn = knn_regressor.predict(X_test)

# 计算均方误差（MSE）和决定系数（R²）
mse_ann = mean_squared_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f'人工神经网络的均方误差 (MSE): {mse_ann:.4f}')
print(f'k近邻模型的均方误差 (MSE): {mse_knn:.4f}')
print(f'人工神经网络的决定系数 (R²): {r2_ann:.4f}')
print(f'k近邻模型的决定系数 (R²): {r2_knn:.4f}')

# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))

# 人工神经网络模型
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ann, color='blue', label='预测值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='实际值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('人工神经网络回归模型')
plt.legend()

# k近邻模型
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_knn, color='green', label='预测值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='实际值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('k近邻回归模型')
plt.legend()

plt.tight_layout()
plt.show()