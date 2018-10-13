import torch as t
import numpy as np

np_data = np.arange(6).reshape((2,3))

t_data = t.from_numpy(np_data)
t_np_data = t_data.numpy()
# print 'numpy\n', np_data, '\ntorch', t_data, 't_np_data \n', t_np_data

data = [-1, 2, -3, 4]
tensor = t.FloatTensor(data)
# print '\ntensor', tensor

t_abs = t.abs(tensor)
# print 't_abs', t_abs

t_mean = t.mean(tensor)
# print 't_mean', t_mean

data = [[1, 2], [3, 4]]
tensor = t.FloatTensor(data)
# print t.mm(tensor,tensor)
# print tensor.dot(tensor)







