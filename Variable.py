import torch as t
from torch.autograd import Variable

tensor = t.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

# print 'tensor', tensor
# print 'variable', variable

t_out = t.mean(tensor*tensor)
v_out = t.mean(variable*variable)

# print 't_out', t_out
# print 'v_out', v_out

v_out.backward()

print variable.grad
print variable.data
print variable.data.numpy()
