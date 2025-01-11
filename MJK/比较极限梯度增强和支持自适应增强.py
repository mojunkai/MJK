import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 步骤 1：导入数据
data = pd.read_csv('housing.csv')

# 假设数据集中的目标变量名为 'median_house_value'，其余为特征
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# 步骤 2：处理分类变量
# 使用独热编码处理分类变量
X = pd.get_dummies(X, drop_first=True)

# 步骤 3：处理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 步骤 4：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤 5：训练自适应增强回归模型
ada = AdaBoostRegressor(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# 步骤 6：训练XGBoost回归模型
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)

# 步骤 7：在测试集上进行预测
y_pred_ada = ada.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# 步骤 8：计算评估指标
# 自适应增强回归模型
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)
print(f"自适应增强回归模型 - 均方误差（MSE）: {mse_ada}")
print(f"自适应增强回归模型 - 决定系数（R²）: {r2_ada}")

# XGBoost模型
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost模型 - 均方误差（MSE）: {mse_xgb}")
print(f"XGBoost模型 - 决定系数（R²）: {r2_xgb}")

# 步骤 9：绘制实际值与预测值的对比图
plt.figure(figsize=(12, 6))

# 自适应增强回归模型
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ada, color='blue', label='AdaBoost Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('AdaBoost - 实际值与预测值的对比图')
plt.legend()

# XGBoost模型
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, color='green', label='XGB Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('XGB - 实际值与预测值的对比图')
plt.legend()

plt.tight_layout()
plt.show()



