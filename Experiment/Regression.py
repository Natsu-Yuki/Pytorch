import torch as t
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt

x = t.unsqueeze(t.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*t.rand(x.size())

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())


class Net(t.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = t.nn.Linear(n_features,n_hidden)
        self.predi = t.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = self.predi(x)
        return x


net = Net(1, 10, 1)

optimizer = t.optim.SGD(net.parameters(), lr=0.2)
loss_func = t.nn.MSELoss()

plt.ion()
# print y.size()
for t in range(1000):
    predi = net(x)
    loss = loss_func(predi, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%50 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predi.data.numpy(), color='red')
        plt.title('range:{},loss:{}'.format(t, loss.data.numpy()))

        plt.pause(0.1)

plt.ioff()
plt.show()

