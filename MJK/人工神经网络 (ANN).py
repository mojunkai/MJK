# ANN.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标标签

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'ANN 准确率: {accuracy:.4f}')

# 绘制准确率柱状图
plt.figure(figsize=(6, 4))
plt.bar(['ANN'], [accuracy], color='skyblue')
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('ANN 模型的准确率')
plt.show()