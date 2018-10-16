import torch as t
from torch.autograd import Variable
import numpy as np
from timeit import timeit

tensor = t.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

# print 'tensor', tensor
# print 'variable', variable

t_out = t.mean(tensor*tensor)
v_out = t.mean(variable*variable)

# print 't_out', t_out
# print 'v_out', v_out

v_out.backward()

# print (variable.grad)
# print (variable.data)
# print (variable.data.numpy())

a = t.Tensor(2, 3)
b = a.tolist()
c = a.numpy()
# print(type(b), b)
# print(type(c), c)
# print(type(a.size()), a.size(), a.size()[0])

# print(a.nelement())

# print(type(t.arange(1, 10, 1)))
# print(t.linspace(1, 10, 10))

a = t.arange(0, 6, 1)
b = a.view(2, 3)
# print(b)
c = b.unsqueeze(0)
# print(c, b.unsqueeze(1))
# print(b.squeeze(-1))
# print(a.unsqueeze(1))

a = t.arange(0, 27, 1).view(3, 3, 3)
# print(a)
# print(a[0, 0, 0])

# t.set_default_tensor_type('torch.FloatTensor')
a = t.Tensor(2, 3)
b = a.new(2, 3)
# print(type(a), b)

a = t.arange(0, 12, 1).view(3, 4)
# print(a)
# print(a.sum(0))
# print(a.max(1))
# print(a.numpy(), a.t())

a = np.ones(shape=[2,3])
# print(a,type(a))
b = t.from_numpy(a)
# print(b)
b = t.Tensor(a)
# print(b)

a = t.arange(0,6)
b = a.view(2,3)
# print(a.storage())
# print(b.storage())
# print(id(a.storage())==(id(b.storage())))
a[0] = 233
# print(b)
c = a[2:]
# print(c.storage())
# print(id(a.storage())==(id(c.storage())))

# print(c.data_ptr(), a.data_ptr())


timeit('x=1')
timeit('x=1', number=1)
timeit('[i for i in range(10000)]', number=1)
print(timeit('[i for i in range(100) if i%2==0]', number=10000))
