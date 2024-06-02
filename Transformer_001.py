import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, matthews_corrcoef
import tushare as ts  # 导入 tushare 获取金融数据
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import math

# 设置设备为 GPU（如果可用），否则为 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用 token 初始化 tushare
ts.set_token('6619c8ca3f0e3445679796979720fde9a360d5cb53c3f035')
pro = ts.pro_api()  # 创建 pro API 对象

# 获取给定股票代码和日期范围的每日股票数据
df = pro.daily(ts_code='002230.SZ', start_date='20200106', end_date='20231106')
df.sort_values(by='trade_date', ascending=True, inplace=True)  # 按交易日期升序排序数据

# 从数据框中提取第 8 和第 9 列
eighth_ninth_columns_df = df.iloc[:, [8, 9]]

# 定义模型的超参数
seq_len = 35
num_epochs = 1500
batch_size = 100
input_dim = 2
hidden_dim = 512
num_layers = 6
num_heads = 32
dropout_rate = 0.1
study_rate = 0.000005

# 创建用于时间序列数据的分类数据集的函数
def create_classification_dataset(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - 1):
        sequence = data[i:(i + seq_len)]  # 提取长度为 `seq_len` 的序列
        two_day_change = data[i + seq_len, 0] + data[i + seq_len + 1, 0]  # 计算两天变化的总和
        label = 1 if two_day_change > 0 else 0  # 根据变化分配标签
        X.append(sequence)  # 将序列添加到输入数据中
        Y.append(label)  # 将标签添加到输出数据中
    return np.array(X), np.array(Y)

# 将数据框转换为 numpy 数组并创建分类数据集
data = eighth_ninth_columns_df.values.astype(float)
X, Y = create_classification_dataset(data, seq_len)

# 使用 Min-Max 标准化序列的函数
def normalize_sequences(sequences):
    normalized_sequences = []
    mm = MinMaxScaler()
    for seq in sequences:
        normalized_seq = mm.fit_transform(seq)  # 标准化每个序列
        normalized_sequences.append(normalized_seq)
    return np.array(normalized_sequences)

# 标准化序列
X_normalized = normalize_sequences(X)

# 将数据分割为训练集和测试集的函数
def split_data(x, y, split_ratio):
    train_size = int(len(y) * split_ratio)  # 计算训练集大小
    x_train = x[:train_size]  # 训练集输入数据
    y_train = y[:train_size]  # 训练集标签
    x_test = x[train_size:]  # 测试集输入数据
    y_test = y[train_size:]  # 测试集标签
    return x_train, y_train, x_test, y_test

# 分割数据为训练集和测试集
X_train, Y_train, X_test, Y_test = split_data(X_normalized, Y, 0.7)

# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
# Transformer 分类模型类
class TransformerClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout_rate, max_len=5000):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim, dropout=dropout_rate, 
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, src):
        src = src.view(src.size(0), src.size(1), -1)
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        out = self.fc(memory)
        out = out[:, -1, :]
        return torch.sigmoid(out).squeeze(-1)

# 初始化模型
model = TransformerClassificationModel(input_dim, hidden_dim, num_layers, num_heads, dropout_rate)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=study_rate)

# 将训练和测试数据转换为 tensor 并移动到设备
X_train_tensor = torch.tensor(X_train).float().to(device)
Y_train_tensor = torch.tensor(Y_train).float().to(device)
X_test_tensor = torch.tensor(X_test).float().to(device)
Y_test_tensor = torch.tensor(Y_test).float().to(device)

model.train()  # 设置模型为训练模式

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        Y_batch = Y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()  # 梯度清零
        output = model(X_batch)  # 模型前向传播
        loss = criterion(output.squeeze(), Y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累积损失

    # 每 10 个 epoch 输出一次损失
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / (len(X_train_tensor) // batch_size)))

model.eval()  # 设置模型为评估模式

# 评估模型
with torch.no_grad():
    Y_pred_train_binary = (model(X_train_tensor) > 0.55).to('cpu').numpy()  # 训练集预测
    Y_pred_test_binary = (model(X_test_tensor) > 0.55).to('cpu').numpy()  # 测试集预测

    # 计算训练集上预测为上升的准确率
    correct_rise_predictions_train = np.sum((Y_pred_train_binary == 1) & (Y_train == 1))
    total_rise_predictions_train = np.sum(Y_pred_train_binary == 1)
    modified_accuracy_train = correct_rise_predictions_train / total_rise_predictions_train if total_rise_predictions_train > 0 else 0

    # 计算测试集上预测为上升的准确率
    correct_rise_predictions_test = np.sum((Y_pred_test_binary == 1) & (Y_test == 1))
    total_rise_predictions_test = np.sum(Y_pred_test_binary == 1)
    modified_accuracy_test = correct_rise_predictions_test / total_rise_predictions_test if total_rise_predictions_test > 0 else 0

    print(f'Test Accuracy (Rise Predictions): {modified_accuracy_test:.4f}')

# 获取训练集和测试集的预测结果（四舍五入）
Y_pred_train = model(X_train_tensor).round()
Y_pred_test = model(X_test_tensor).round()

# 将 tensor 转换为 numpy 数组
Y_pred_train_np = Y_pred_train.detach().to('cpu').numpy()
Y_pred_test_np = Y_pred_test.detach().to('cpu').numpy()

Y_train_np = Y_train
Y_test_np = Y_test

# 绘制测试集上的实际值和预测值的比较图
plt.figure(figsize=(7, 5))
plt.plot(Y_test_np, 'blue', label='True Values')
plt.plot(Y_pred_test_np, 'red', linestyle='dashed', label='Predicted Values')
plt.title('Comparison on Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Classification Value')
plt.legend()
plt.show()
