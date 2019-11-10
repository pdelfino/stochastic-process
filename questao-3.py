
import random
import matplotlib.pyplot as plt

def martingal():
    for j in range(10):
        a = 0
        m0 = 1
        v = []
        for i in range(100):
            a = random.random()
            if a > 0.5:
                m0 = m0*(1/2)
            else:
                m0 = m0*(3/2)
            v.append(m0)
        plt.plot(v)
        plt.axis([0, 100, 0, 0.5])
        print(m0)
    plt.show()

martingal()

