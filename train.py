import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
# 定义数据集类

class RNNModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2, dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

from ncltloader import SequenceLoader
# 加载数据
npz_file_path = "./train.npz"
test_data = "./test.npz"
# 超参数
sequence_length = 10
batch_size = 128
learning_rate = 0.001
epochs = 5
device = "cuda"
# 数据加载
train_dataset = SequenceLoader(npz_file_path, 10,1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SequenceLoader('val.npz',10,30)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


val_dataset = SequenceLoader('val.npz',10,1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = RNNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
trainloss=[]
valloss=[]
# 训练
for epoch in range(epochs):
    epoch_loss = 0
    print(f"Epoch {epoch}/{epochs}")
    model.train()

    # 使用 tqdm 包裹 train_loader
    for x, y in tqdm(train_loader, desc="Training", leave=True):
        x = x.to(device) # 添加最后一维作为输入特征维度
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)/batch_size:.4f}")
    trainloss.append(epoch_loss / len(train_loader)/batch_size)
    model.eval()
    epoch_loss = 0

    # 使用 tqdm 包裹 train_loader
    for x, y in tqdm(val_loader, desc="Val", leave=True):
        x = x.to(device) # 添加最后一维作为输入特征维度
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(val_loader)/batch_size:.4f}")
    valloss.append(epoch_loss / len(val_loader)/batch_size)

# 测试并预测
# 测试并预测，使用 MC Dropout
model.eval()
predictions = []
sigmas = []
errors = []
num_mc_samples=50
guard = 0
with torch.no_grad():
    # for x, y in test_loader:
    for x, y in tqdm(test_loader, desc="test", leave=True):
        guard+=1
        if guard>50:
            break
        x = x.to(device)  # 添加最后一维作为输入特征维度
        prediction = []
        sigma = []
        error = []
        input_seq = x.clone()

        for i in range(30):
            mc_outputs = []
            for _ in range(num_mc_samples):
                model.train()  # 激活 Dropout
                mc_output = model(input_seq)
                mc_outputs.append(mc_output.cpu().numpy())

            # 拟合高斯分布
            mc_outputs = np.array(mc_outputs)

            # 求 50 个坐标的平均值
            mean_coords = np.mean(mc_outputs, axis=(0, 1))  # 沿第 0 和第 1 轴求均值

            # print("平均值 (2):", mean_coords)
            # mean = mc_outputs.mean()
            std =  np.std(mc_outputs, axis=(0, 1))
            prediction.append(mean_coords)
            sigma.append(np.sqrt(std[0]**2 + std[1]**2))
            rmse = np.sqrt(np.mean((mean_coords - y[0, i].cpu().numpy()) ** 2))
            error.append(rmse)

            # 更新输入序列，移除第一个值并添加高斯均值
            # input_seq = torch.cat((input_seq[:, 1:, :], torch.tensor(mean).unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=1)
            input_seq = torch.cat(
            (input_seq[:, 1:, :], torch.tensor(mean_coords, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)),
            dim=1
        )

        # predictions.append(prediction)
        # # print(prediction)
        # # print(y[0].item())
        sigmas.append(sigma)
        errors.append(error)

# 保存预测和不确定性（标准差）
# npz_output_path = "./predictions_and_sigmas.npz"
# np.savez(npz_output_path, predictions=predictions, sigmas=sigmas, errors=errors)
# print(f"Predictions, sigmas, and errors saved at: {npz_output_path}")

# 可视化误差和不确定性
mean_errors = np.mean(errors, axis=0)
mean_sigmas = np.mean(sigmas, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(range(30), mean_errors, marker="o", label="Mean Error")
print(mean_errors)
print(mean_sigmas)

plt.fill_between(range(30), mean_errors - 3 * mean_sigmas, mean_errors + 3 * mean_sigmas, color="gray", alpha=0.2, label="3-Sigma Interval")
plt.title("Mean Error with Uncertainty (3-Sigma)")
plt.xlabel("Time Steps")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()
print(trainloss)
print(valloss)