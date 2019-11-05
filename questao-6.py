
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np


# In[59]:


def brownian_motion(M, n, T): #M é o número de caminhos, n é o número de incrementos, T é o tempo final
    h = T/n
    Z = np.sqrt(h) * np.random.normal(size = (M, N-1))
    B = np.zeros(shape = (M, N))
    B[:, 1:] = np.cumsum(Z, axis = 1)
    m = np.max(B, axis = 1)
    
    plt.plot(np.linspace(0, 1, N), B[:10, :].T)
    plt.show()
    x = np.linspace(0, 3.5, 100)
    y = ((2/(np.pi))**(1/2))*(np.exp((-x**2)/(2)))
    plt.plot(x, y)
    plt.hist(m, normed = True)


# In[61]:


brownian_motion(1000, 100, 1)

