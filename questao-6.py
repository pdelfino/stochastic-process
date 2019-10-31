
# coding: utf-8

# In[99]:


import matplotlib.pyplot as plt
import numpy as np


# In[124]:


def brownian_motion(n, T): #n é o número de intervalos, T é o tempo final
    h = T/n
    X = []
    Y = []
    max = 0
    B = 0
    for i in range(n):
        Z = np.random.normal(0,h)
        Y.append(B + Z)
        X.append(i)
        if max < B + Z:
            max = B + Z
    plt.plot(X, Y)
    plt.show()
    return max


# In[125]:


brownian_motion(100, 1)


# In[122]:


hist = []
for i in range(10000):
    hist.append(brownian_motion(100, 1))


# In[123]:


plt.hist(hist)
plt.show


# In[108]:


x = np.linspace(-5,5,100)
y = ((2/(np.pi))**(1/2))*(np.exp((-x**2)/(2)))
plt.plot(x, y)
plt.show

