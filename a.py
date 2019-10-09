import numpy as np
a=np.random.rand(2,3,4)
print(a.shape)
b=a[...,0:1]
c=a[...,0]
print(b.shape)
print(c.shape)