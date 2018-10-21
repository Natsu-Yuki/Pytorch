from torch import nn
import torch as t
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.image as img
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
np.random.seed(0)

class Linear(nn.Module):
    def __init__(self, in_, out_):
        super(Linear, self).__init__()
        self.w = nn.Parameter(t.randn(in_,out_))
        self.b = nn.Parameter(t.randn(out_))

    def forward(self, x):
        x = x.mm(self.w)
        return x+self.b.expand_as(x)


layer = Linear(4, 3)
in_ = Variable(t.randn(2, 4))
out = layer(in_)

image = img.imread(r'image/a.jpg')
plt.imshow(image)
# plt.title('shape:{}'.format(image.shape))
# plt.show()
t_i = t.Tensor(image)

i = Variable(t.randn(2,3))
linear = Linear(3,4)
h = linear(i)
print(h)
bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4)*4
bn.bias.data = t.zeros(4)

bn_out = bn(h)
print(bn_out)

net = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=2, stride=1),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

i = Variable(t.randn(2, 3, 4))

lstm = nn.LSTM()