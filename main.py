import paddle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import paddle.nn.functional as F

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
# matplotlib.use('TkAgg')
# sns.pairplot(df.dropna(), y_vars=feature_names[-1], x_vars=feature_names[::-1],diag_kind='kde')
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

#将训练数据集和测试数据集按照8:2的比例分开
ratio = 0.8
offset = int(housing_data.shape[0] * ratio)
train_data = housing_data[:offset]
test_data = housing_data[offset:]


# 模型组网
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(13, 1,)
    def forward(self, inputs):
        pred = self.fc(inputs)
        return pred

# 定义训练过程中损失值变化趋势的方法
train_nums = []
train_costs = []

def draw_train_process(iters, train_costs):
    plt.title("training cost", fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.show()

# 训练模型
y_preds = []
labels_list = []

def train(model):
    print('start training ...')
    # 开启模型训练模式
    model.train()
    EPOCH_NUM = 500
    train_num = 0
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含20条数据
        mini_batches = [train_data[k: k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for batch_id, data in enumerate(mini_batches):
            features_np = np.array(data[:, :13], np.float32)
            labels_np = np.array(data[:, -1:], np.float32)
            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)
            # 向前计算
            y_preds = model(features)
            cost = F.mse_loss(y_preds, label = labels)
            train_cost = float(cost)
            # 反向传播
            cost.backward()
            # 最小化loss 更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()

            if batch_id%30 == 0 and epoch_id%50 == 0:
                print("Pass:%d,Cost:%0.5f"%(epoch_id, train_cost))
            train_num = train_num + BATCH_SIZE
            train_nums.append(train_num)
            train_costs.append(train_cost)

model = Regressor()
train(model)

matplotlib.use('TkAgg')
draw_train_process(train_nums, train_costs)
