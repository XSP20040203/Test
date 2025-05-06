import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # 2D可视化不需要3D工具
import matplotlib.animation as animation
from pylab import mpl # 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 为了结果的可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成合成数据 (沿用之前的例子)
true_w = 2.5
true_b = 5.0
num_samples = 200

# 生成 x 值 (一维输入)
X_np = np.random.rand(num_samples, 1) * 20 - 10 # 生成 -10 到 10 之间的随机 x 值

# 生成 y 值，加入噪声
y_np = true_w * X_np + true_b + np.random.randn(num_samples, 1) * 2 # 加入一些噪声

# 将 NumPy 数组转换为 PyTorch 张量
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# 2. 定义线性模型 (一维输入，一维输出)
model = nn.Linear(1, 1)

# 3. 定义损失函数
criterion = nn.MSELoss()

# 4. 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- 训练模型并记录参数历史 ---
num_epochs = 500 # 减少 epoch 数量以获得更流畅的动画
param_history = [] # 用于保存每一轮的 (w, b) 值
loss_history = [] # 用于记录损失值 (虽然动画不直接用，保留着也行)

print("开始训练并记录参数历史...")
for epoch in range(num_epochs):
    # 记录当前参数值 (在 optimizer.step() 之前记录当前步的值)
    current_w = model.weight.data.item()
    current_b = model.bias.data.item()
    param_history.append((current_w, current_b))

    # 前向传播：计算预测值
    y_predicted = model(X_tensor)

    # 计算损失
    loss = criterion(y_predicted, y_tensor)
    loss_history.append(loss.item())

    # 反向传播：计算梯度
    optimizer.zero_grad()
    loss.backward()

    # 参数更新：根据梯度和学习率更新参数
    optimizer.step()

    if (epoch + 1) % 100 == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练结束，参数历史记录完毕。")

# --- 动态模型拟合可视化 ---

# 创建图表和轴
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制原始数据点 (静态不变)
ax.scatter(X_np, y_np, label='原始数据', alpha=0.6, s=20)

# 初始化要动态更新的直线
plot_x = np.array([X_np.min(), X_np.max()]).reshape(-1, 1)
initial_w, initial_b = param_history[0]
initial_y = initial_w * plot_x + initial_b
line, = ax.plot(plot_x, initial_y, color='red', linewidth=2, label='拟合直线') # 注意这里的逗号，line 是一个列表

# 设置图表标签和标题
ax.set_xlabel('X')
ax.set_ylabel('y')
# 设置初始标题，动画更新时可能会覆盖
ax.set_title('动态模型拟合过程')
ax.legend()
ax.grid(True)

# 确保轴的范围固定，避免动画过程中抖动
ax.set_xlim(X_np.min() - 1, X_np.max() + 1)
ax.set_ylim(y_np.min() - 5, y_np.max() + 5)

# 也可以获取标题 Text 对象，然后在 update 中直接修改其文本
title_artist = ax.set_title(f'动态模型拟合过程\nEpoch 1/{num_epochs}, w={initial_w:.4f}, b={initial_b:.4f}')


# 动画更新函数
def update(frame):
    # frame 是当前帧的索引 (从 0 到 num_epochs-1)
    current_w, current_b = param_history[frame]

    # 使用当前帧的参数计算直线新的 y 值
    current_y = current_w * plot_x + current_b

    # 更新直线的数据
    line.set_data(plot_x.flatten(), current_y.flatten())

    # 更新标题文本 (如果需要动态显示 epoch 和参数)
    title_artist.set_text(f'动态模型拟合过程\nEpoch {frame+1}/{num_epochs}, w={current_w:.4f}, b={current_b:.4f}')


    # 返回一个包含所有更新的 Artist 对象的序列
    # 在 blit=True 时，只返回实际修改了数据的 Artists (主要是 line)
    # 返回 line, 即可 (逗号表示返回一个包含 line 的元组)
    return line, title_artist # 尝试返回 line 和 title_artist

# 创建动画对象
# fig: 动画所在的 Figure 对象
# update: 动画更新函数
# frames: 动画的总帧数 (等于参数历史记录的数量)
# interval: 每帧之间的毫秒延迟
# blit=True: 开启 blitting 优化，只重绘变化的部分
# repeat=False: 动画只播放一次
try:
    print("尝试使用 blit=True 运行动画...")
    ani = animation.FuncAnimation(fig, update, frames=num_epochs, interval=50, blit=True, repeat=False)
except RuntimeError as e:
    print(f"blit=True 运行时出错: {e}")
    print("尝试使用 blit=False 运行动画...")
    # 如果 blit=True 出错，尝试 blit=False
    ani = animation.FuncAnimation(fig, update, frames=num_epochs, interval=50, blit=False, repeat=False)

# 显示动画
plt.show()
