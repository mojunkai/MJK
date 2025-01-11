# LR.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据集
cancer = datasets.load_breast_cancer()
X = cancer.data  # 特征数据
y = cancer.target  # 目标标签

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'LR 准确率: {accuracy:.4f}')

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
importances = np.abs(model.coef_[0])
plt.bar(range(X.shape[1]), importances, color='skyblue')
plt.title('LR 特征重要性')
plt.xlabel('特征索引')
plt.ylabel('重要性')
plt.show()

# 计算偏差和方差
scores = cross_val_score(model, X, y, cv=10)
bias = (1 - scores.mean()) ** 2
variance = scores.var()
print(f'偏差: {bias:.4f}')
print(f'方差: {variance:.4f}')

# 绘制偏差和方差图
plt.figure(figsize=(10, 6))
plt.bar(['偏差', '方差'], [bias, variance], color=['blue', 'orange'])
plt.title('LR 偏差和方差')
plt.ylabel('值')
plt.show()