from model import MNIST_MLP
from utils import load_data
import matplotlib.pyplot as plt

# 定义不同的学习率
learning_rates = [0.001, 0.01, 0.1]
train_losses_all = []

# 加载数据
train_data, test_data = load_data()

# 定义训练轮次
num_epochs = 5  # 这里可以修改为你想要的训练轮次

# 针对每个学习率进行训练
for lr in learning_rates:
    # 创建模型，并设置训练轮次
    mlp = MNIST_MLP(lr=lr, max_epoch=num_epochs)
    mlp.build_model()
    mlp.init_model()

    # 训练模型
    mlp.train(train_data)

    # 记录训练损失
    train_losses_all.append(mlp.train_losses)

    # 评估模型
    mlp.evaluate(test_data)

# 可视化不同学习率下的训练损失
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(train_losses_all[i], label=f'Learning Rate = {lr}')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('Effect of Learning Rate on Convergence Speed')
plt.legend()
plt.savefig('learning_rate_sensitivity.png')
plt.show()