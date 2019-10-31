
# coding: utf-8

# In[517]:


import numpy as np
import random


# In[604]:


def questao4_1():
    s = 0
    Tn = []
    x = 0
    while s + x < 5:
        s += x
        Tn.append(x)
        x = np.random.exponential(1)
    Tn.append(5-s) 
    
    #print(Tn)
    
    Nt =  0
    for i in range(1, len(Tn)-1):
        Nt += i*Tn[i]
        
    return Nt


# In[605]:


questao4_1()


# In[606]:


s=0
for i in range(100000):
    s+=questao4_1()
s/100000


# In[764]:


def questao4_2():
    n = np.random.poisson(5)
    Tn = []
    for i in range(0, n):
        Tn.append(np.random.uniform(0., 5.))
    Tn.sort()
    Tn.append(5)
    Nt = 0
    for i in range(1, len(Tn)):
        Nt += (i)*(Tn[i]-Tn[i-1])       
    return Nt


# In[765]:


questao4_2()


# In[766]:


s=0
for i in range(100000):
    s+=questao4_2()
s/100000

