
# coding: utf-8

# In[139]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[140]:


def questao4_1():
    s = 0
    Xn = []
    Tn = []
    x = 0
    while s + x < 5:
        s += x
        Xn.append(x)
        x = np.random.exponential(1)
        Tn.append(s)
    Xn.append(5-s)
    Tn.append(5)
    
    #print(Tn)
    
    It =  0
    Nt = [0]
    for i in range(1, len(Xn)-1):
        Nt.append(i)
        It += i*Xn[i]
        
    return Xn, Nt, Tn, It


# In[163]:


Xn, Nt, Tn, It = questao4_1()


# In[164]:


Nt, Tn


# In[36]:


plt.step(Tn, Nt + [Nt[-1]], where='post')


# In[40]:


s=0
for i in range(100000):
    _, _, _, It = questao4_1()
    s += It
s/100000


# In[170]:


def questao4_2():
    n = np.random.poisson(5)
    Tn = []
    for i in range(0, n):
        Tn.append(np.random.uniform(0., 5.))
    Tn.sort()
    Tn.append(5)
    It = 0
    Nt = [0]
    for i in range(1, len(Tn)):
        It += (i)*(Tn[i]-Tn[i-1]) 
        if i != len(Tn)-1:
            Nt.append(i)
    return It, Nt, Tn


# In[171]:


In, Nt, Tn = questao4_2()


# In[172]:


Nt, Tn


# In[173]:


plt.step(Tn, Nt + [Nt[-1]], where='post')


# In[174]:


s=0
for i in range(100000):
    It, _, _, = questao4_2()
    s += It
s/100000

