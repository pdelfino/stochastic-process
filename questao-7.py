
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt


# In[34]:


def black_scholes(n): #n é o número de valores S gerados
    W = np.random.normal(size = n)
    S = np.zeros(shape = n)
    for i in range(n):
        S[i] = 100*np.exp(-0.055+ W[i]*0.04)
    K = np.linspace(80, 120, 9)
    C = np.zeros(shape = 9)
    for k in range(9):
        for i in S:
            if i > K[k]:
                C[k] += np.exp(-0.05)*(i - K[k])
        C[k] = C[k]/n
        
    plt.plot(K, C, 'bo')
    plt.show


# In[35]:


black_scholes(100)

