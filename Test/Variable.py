import torch as t
from torch.autograd import Variable as V

a = V(t.ones(3, 4), requires_grad=True)

# print(a, type(a))
# print(a.data, type(a.data))

b = V(t.zeros(3, 4), requires_grad=True)
c = a+b
# print(c)
d = c.sum()
d.backward()
# print(a.grad)
d = c.sum()
d.backward()
# print(a.grad)


def f(x):
    return x**2*t.exp(x)


def gradf(x):
    return 2*x*t.exp(x)+x**2*t.exp(x)


x = V(t.randn(3, 4), requires_grad=True)
y = f(x)
# print(y)

y.backward(t.ones(y.size()))
# print(x.grad)
# print(gradf(x))


