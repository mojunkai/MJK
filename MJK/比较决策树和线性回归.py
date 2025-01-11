# DecisionTreeRegression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()
X = diabetes.data  # 特征数据
y = diabetes.target  # 目标标签

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树回归模型
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# 训练线性回归模型
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_dt = dt_regressor.predict(X_test)
y_pred_lr = lr_regressor.predict(X_test)

# 计算均方误差（MSE）和决定系数（R²）
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'决策树回归模型的均方误差 (MSE): {mse_dt:.4f}')
print(f'决策树回归模型的决定系数 (R²): {r2_dt:.4f}')
print(f'线性回归模型的均方误差 (MSE): {mse_lr:.4f}')
print(f'线性回归模型的决定系数 (R²): {r2_lr:.4f}')

# 绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))

# 决策树回归模型
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_dt, color='blue', label='预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label='实际值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('决策树回归模型')
plt.legend()

# 线性回归模型
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lr, color='green', label='预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label='实际值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('线性回归模型')
plt.legend()

plt.tight_layout()
plt.show()