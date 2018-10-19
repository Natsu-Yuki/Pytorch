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

# print(x.requires_grad, b.requires_grad, w.requires_grad)
# print(x.is_leaf,w.is_leaf,b.is_leaf, y.is_leaf, z.is_leaf)
# print(z.grad_fn)


def abs(x):
    if x.data[0]>0: return x
    else:return -x


x = Variable(t.ones(1),requires_grad=True)
y = abs(x)
y.backward()
# print(x.grad)

x = Variable(-1*t.ones(1),requires_grad=True)
y = abs(x)
y.backward()
# print(x.grad)


# def f(x):
#     result = 1
#     for i in x:
#         if i.data[0] > 0 : result = i*result
#     return result
#
#
# t.set_default_tensor_type("torch.FloatTensor")
#
# x = Variable(t.arange(-2, 4).float(), requires_grad=True)
#
# y = f(x)
#
# y.backward()

# print(x.grad)

x = Variable(t.ones(2,3))
w = Variable(t.rand(2,3), requires_grad=True)
y = x*w
print(x.requires_grad, w.requires_grad, y.requires_grad)



x = Variable(t.ones(2,3),volatile = True)
w = Variable(t.rand(2,3), requires_grad=True)
y = x*w
print(x.requires_grad, w.requires_grad, y.requires_grad)