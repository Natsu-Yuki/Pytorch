import torch as t
from torch.autograd import Variable

x = Variable(t.ones(2,3))
b = Variable(t.rand(2,3), requires_grad=True)
w = Variable(t.rand(2,3), requires_grad=True)
y = w*x
z = y+b

# print(x)
# print(w)
# print(y)

print(x.requires_grad, b.requires_grad, w.requires_grad)
print(x.is_leaf,w.is_leaf,b.is_leaf, y.is_leaf, z.is_leaf)

print(z.grad_fn)