import torch as t
import torch.nn as nn
from torch.autograd import Variable

x = Variable(t.randn(10, 1, 28, 28))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2

            ),
            # nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.conv2d(x)


n = Net()
out = n(x)

print(out.size())