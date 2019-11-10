import numpy as np
import matplotlib.pyplot as plt

def black_scholes(n): #n é o número de valores S gerados
    W = np.random.normal(size = n)
    S = 100*np.exp(-0.055 + W*0.04)
    K = np.linspace(80, 120, 9)
    C = np.zeros(shape = 9)
    for k in range(9):
        C[k] = np.mean(np.exp(-0.05) * np.maximum(S - K[k], 0.0))
    
    plt.plot(K, C)
    plt.show()

print (black_scholes(100))

