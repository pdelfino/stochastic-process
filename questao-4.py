import numpy as np
import random
import matplotlib.pyplot as plt

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
    
    It =  0
    Nt = [0]
    for i in range(1, len(Xn)-1):
        Nt.append(i)
        It += i*Xn[i]
        
    return Xn, Nt, Tn, It

Xn, Nt, Tn, It = questao4_1()

#print (Nt, Tn)

plt.step(Tn, Nt + [Nt[-1]], where='post')

s=0
for i in range(100000):
    _, _, _, It = questao4_1()
    s += It

print ("s/100000 no tópico (i): ",s/100000)

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

In, Nt, Tn = questao4_2()

#print (Nt, Tn)

plt.step(Tn, Nt + [Nt[-1]], where='post')

s=0
for i in range(100000):
    It, _, _, = questao4_2()
    s += It

print ("s/100000 no tópico (ii):",s/100000)

plt.show()
