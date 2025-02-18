import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一个简单的三层多层感知器（MLP）模型作为 Q 网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 输出层有两个动作：点击和不点击
        self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)
        if self.fc3.bias is not None:
            nn.init.constant_(self.fc3.bias, 0)

# 超参数
gamma = 0.95
learning_rate = 0.0001  # 降低学习率
num_epochs = 5000  # 增加训练轮数
batch_size = 16  # 减少批次大小
penalty_factor = 0.001  # 保持惩罚因子不变

# 创建 Q 网络和目标 Q 网络
q_network = MLP().to(device)
target_q_network = MLP().to(device)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 尝试加载现有的 PyTorch 模型
pth_model_path = r"E:\steam\steamapps\common\Stardew Valley\Mods\RL-fishing\assets\policy_net.pth"
if os.path.exists(pth_model_path):
    print(f"加载现有的 PyTorch 模型: {pth_model_path}")
    q_network.load_state_dict(torch.load(pth_model_path, map_location=device, weights_only=True))
    target_q_network.load_state_dict(q_network.state_dict())
else:
    print("未找到现有的 PyTorch 模型，从头开始训练")
    q_network._initialize_weights()
    target_q_network._initialize_weights()

# 读取 replayMemory.csv 文件中的数据
data = pd.read_csv(r"E:\steam\steamapps\common\Stardew Valley\replayMemory.csv", header=None)
states = data.iloc[:, :3].values
next_states = data.iloc[:, 3:6].values
rewards = data.iloc[:, 6].values
actions = data.iloc[:, 7].values

# 记录每个 epoch 的 q-value
q_values_list = []

# 训练 DQN
for epoch in range(num_epochs):
    # 随机抽取一个批次的数据
    indices = np.random.choice(len(states), batch_size)
    batch_states = torch.tensor(states[indices], dtype=torch.float32).to(device)
    batch_next_states = torch.tensor(next_states[indices], dtype=torch.float32).to(device)
    batch_rewards = torch.tensor(rewards[indices], dtype=torch.float32).to(device)
    batch_actions = torch.tensor(actions[indices], dtype=torch.int64).to(device)

    # 计算当前 Q 值
    q_values = q_network(batch_states)
    q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

    # 记录 q-value
    q_values_list.append(q_values.mean().item())

    # 计算目标 Q 值
    with torch.no_grad():
        next_q_values = target_q_network(batch_next_states).max(1)[0]
        target_q_values = batch_rewards + gamma * next_q_values

    # 计算连续采取同一动作的惩罚
    action_changes = (batch_actions[1:] != batch_actions[:-1]).float()
    penalty = penalty_factor * (1 - action_changes).sum()

    # 计算损失并更新 Q 网络
    loss = criterion(q_values, target_q_values) + penalty
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每隔一段时间更新目标 Q 网络
    if epoch % 5 == 0:  # 增加目标 Q 网络的更新频率
        target_q_network.load_state_dict(q_network.state_dict())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存训练好的 Q 网络为 PyTorch 格式
torch.save(q_network.state_dict(), pth_model_path)
print(f"PyTorch 模型已保存到 {pth_model_path}")

# 导出训练好的 Q 网络为 ONNX 格式
dummy_input = torch.randn(1, 3).to(device)
onnx_model_path = r"E:\steam\steamapps\common\Stardew Valley\Mods\RL-fishing\assets\policy_net.onnx"
torch.onnx.export(q_network, dummy_input, onnx_model_path, 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f"ONNX 模型已保存到 {onnx_model_path}")

# 绘制 q-value 的变化图
plt.plot(q_values_list)
plt.xlabel('Epoch')
plt.ylabel('Q-value')
plt.title('Q-value over Epochs')
plt.show()