import paddle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
datafile = '波士顿房价数据集.csv'
housing_data = pd.read_csv(datafile)
housing_data = np.array(housing_data)
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# feature_num = len(feature_names)
# print(housing_data.shape[0])
# housing_data = housing_data.reshape([housing_data.shape[0] // feature_num, feature_num])
# print('0')
fearture_np = np.array([x[:13] for x in housing_data], np.float32)
labels_np = np.array([x[-1] for x in housing_data], np.float32)

df = pd.DataFrame(housing_data, columns=feature_names)
matplotlib.use('TkAgg')
sns.pairplot(df.dropna(), y_vars=feature_names[-1], x_vars=feature_names[::-1],diag_kind='kde')
# plt.show()
# 相关性分析
# fig, ax = plt.subplots(figsize=(15, 1))
# corr_data = df.corr().iloc[-1]
# corr_data = np.asarray(corr_data).reshape(1, 14)
# ax = sns.heatmap(corr_data, cbar=True, annot=True)
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.boxplot(data=df.iloc[:, 0:13])
# plt.show()

feature_max = housing_data.max(axis=0)
feature_min = housing_data.min(axis=0)
feature_avg = housing_data.sum(axis=0) / housing_data.shape[0]

BATCH_SIZE = 20

def feature_norm(input):
    f_size = input.shape
    output_features = np.zeros(f_size, np.float32)
    for batch_id in range(f_size[0]):
        for index in range(13):
            output_features[batch_id][index] = (input[batch_id][index] - feature_avg[index]) / (feature_max[index] - feature_min[index])
    return output_features



fig, ax = plt.subplots(figsize=(10, 8))
housing_features = feature_norm(housing_data[:, :13])
housing_data = np.c_[housing_features, housing_data[:, -1]].astype(np.float32)
feature_np = np.array([x[:13] for x in housing_data], np.float32)
data_np = np.c_[feature_np, labels_np]
df = pd.DataFrame(data_np, columns=feature_names)
ax2 = sns.boxplot(data=df.iloc[:, 0:13])
plt.show()