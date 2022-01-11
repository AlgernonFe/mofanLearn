import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# unsqeeze把一维数据变为二维数据,torch只处理二维数据
y = x.pow(2) + 0.2*torch.rand(x.size())
# x的平方加上随机噪点的影响
x = Variable(x)
y = Variable(y)
# 将x, y都变成variable的形式，交给神经网络处理

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
# 绘制散点图

# 定义Neural Network
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        # 层信息都是属性
        self.hidden = torch.nn.Linear(n_features, n_hidden) # 隐藏层：n_feature-输入数， n_hidden-输出数
        self.predict = torch.nn.Linear(n_hidden, n_output) # 预测层：输入数-n_hidden, 输出数-1

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1) # 1个输入：x， 10个隐藏层， 1个输出
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# 使用SGD优化器，传入神经网络中的参数net.parameters(), lr:学习率learning rate
loss_func = torch.nn.MSELoss()
# MSE：mean square error均方差损失函数，作为分类误差

# 开始训练
for t in range(1000):
    prediction = net(x)

    loss = loss_func(prediction, y) # 预测值在前，真实值在后

    optimizer.zero_grad() # 把所有参数梯度降为零
    loss.backward() # 反向传递variable：loss
    optimizer.step() # 优化梯度
    if t %5 == 0: #每5步打印一次过程
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()




