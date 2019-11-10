# Stochastic Processing   
---

 + Programming assignments for the course "Processos Estocásticos" (Stochastic Process) at the Escola de Mamemática Aplicada (EMAp). 

 + Students: 
   - Pedro Delfino; 
   - Bruna Fistarol; and
   - Danillo Fiorenza.
   
 + Professor: PhD Yuri Saporito

 + Date: 2019.2.

 + Tech Stack: Python 3.6.8.  

    + **Atenção,** a específicação acima é especialmente importante para a função random.choices do pacote random. 

 + Exercises: 

   - see [problem set pdf](https://github.com/pdelfino/stochastic-process/blob/master/problem-set.pdf)

 + Support Material:

   +  [lecture notes](https://drive.google.com/file/d/0BwDJjYFvJgwNZFk1dmFKeExKblU/view) written by Professor Yuri; 

   + Introduction To Stochastic Process With R - Roberto Dobrow.

   
---
   
### Questão 1
   

   
A imagem abaixo ilustra bem as probabilidades de transição. Além disso, é possível perceber que os estados 2 é o único transiente. Todos os outros são recorrentes.  Outra característica evidente é separação em dois grafos, de modo que os estados 1, 2 e 3 não se comunicam com os estados 4 e 5.                                                               : 
   ![](https://github.com/pdelfino/stochastic-process/blob/master/diagrama.png)


A questão exige basicamente que a distribuição estacionária seja encontrada numericamente. Como simulação temos:



```python
   import numpy as np
   import random
   
   P = np.array([[1/3,  0 , 2/3,  0 ,  0 ],
                 [1/4, 1/2, 1/4,  0 ,  0 ],
                 [1/2,  0 , 1/2,  0 ,  0 ],
                 [ 0 ,  0 ,  0 ,  0 ,  1 ],
                 [ 0 ,  0 ,  0 , 2/3, 1/3]])
   
   #print (np.matmul(P,P))
   
   def stationary_state(initial_state,simulation_size):
       states = [0,1,2,3,4]
   
       initial_state_weigth_options = initial_state 
       #print ("initial_state_weigth_options",initial_state_weigth_options)
       #print ("P[",str(initial_state_weigth_options),"]",P[initial_state_weigth_options])
   
       n = simulation_size
       
       for i in range(1,100):
   
           next_state = (random.choices(states, P[initial_state_weigth_options]))
           next_state = next_state[0]
           #print ("next_state", next_state)
   
           initial_state_weigth_options = next_state
           #print ("new initial state", initial_state_weigth_options)
           
           next_state = (random.choices(states,P[initial_state_weigth_options]))
           next_state = next_state[0]
           #print ("next_state", next_state)
           
           initial_state_weigth_options = next_state
   
       final_state = next_state
   
       return ("initial state: ",str(initial_state)," | final state: ",str(final_state))
   
   
   def simulate(initial_state,n):
       dictionary = {"0":0,"1":0,"2":0,"3":0,"4":0}
   
       for i in range(1,n):
           final_state = stationary_state(initial_state,n)
           final_state = final_state[-1]
           dictionary[str(final_state)] += 1/n
   
       return dictionary
   
   print ("Para facilitar o código, chamamos o estado 1 de 0, o 2 de 1, o 3 de 2, o 4 de 3 e o 5 de 4. Assim, o novo nome do estado coincide com seu índice na matriz. Dessa forma, usando a nova nomenclatura, temos:")
   print ("\n")
   
   for i in range(0,5):
       print ("Estado de origem: ",i,"| Distribuição estacionária",simulate(i,10000))
       print ("\n")
   
   print ("Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.")
   
```



O código demora um pouco para rodar, já que são feitas **10 mil simulações**. Em cada simulação, modelamos o problema de modo que ocorreram 100 iterações aleatórias. Isto é, o processo iniciava em uma estado e, de acordo com as probabilidades associadas a cada estado, havia um **sorteio ponderado** para o próximo estado a ser visitado. Após 100 iterações, dava-se o último estado visitado como posição final.



O código retorna este resultado:

```python
   
   Estado de origem:  0 | Distribuição estacionária {'0': 0.4339999999999685, '1': 0, '2': 0.565899999999954, '3': 0, '4': 0}
   
   
   Estado de origem:  1 | Distribuição estacionária {'0': 0.42849999999996913, '1': 0, '2': 0.5713999999999534, '3': 0, '4': 0}
   
   
   Estado de origem:  2 | Distribuição estacionária {'0': 0.4289999999999691, '1': 0, '2': 0.5708999999999534, '3': 0, '4': 0}
   
   
   Estado de origem:  3 | Distribuição estacionária {'0': 0, '1': 0, '2': 0, '3': 0.4077999999999714, '4': 0.5920999999999511}
   
   
   Estado de origem:  4 | Distribuição estacionária {'0': 0, '1': 0, '2': 0, '3': 0.3960999999999727, '4': 0.6037999999999498}
   
   
   Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.
   
```



Portanto, os resultados estão convergindo para os valores corretos.



----

### Questão 2

O Código é 

```python
import numpy as np
import random

P = np.array([[1/3,  0 ,  0 , 2/3],
              [1/4, 1/2, 1/4,  0 ],
              [1/2,  0 , 1/2,  0 ],
              [ 0 , 1/3,  0 , 2/3]])

#print (np.matmul(P,P))

def stationary_state(initial_state,simulation_size):
    states = [0,1,2,3]

    initial_state_weigth_options = initial_state 
    #print ("initial_state_weigth_options",initial_state_weigth_options)
    #print ("P[",str(initial_state_weigth_options),"]",P[initial_state_weigth_options])

    n = simulation_size
    
    for i in range(1,n):

        next_state = (random.choices(states, P[initial_state_weigth_options]))
        next_state = next_state[0]
        #print ("next_state", next_state)

        initial_state_weigth_options = next_state
        #print ("new initial state", initial_state_weigth_options)
        
        next_state = (random.choices(states,P[initial_state_weigth_options]))
        next_state = next_state[0]
        #print ("next_state", next_state)
        
        initial_state_weigth_options = next_state

    final_state = next_state

    return ("initial state: ",str(initial_state)," | final state: ",str(final_state))

#print (stationary_state(0,5))
print ("Para facilitar o código, chamamos o estado 1 de 0, o 2 de 1, o 3 de 2, o 4 de 3 e o 5 de 4. Assim, o novo nome do estado coincide com seu índice na matriz. Dessa forma, usando a nova nomenclatura, temos:")
print ("\n")


total = 0
for i in range(2,1000):
    x_i = int((stationary_state(0,i)[-1]))+1
    #print ("x_i",x_i)
    total += (x_i)**2 
    #print ("total",total/998)
print (total/998)

print ("Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.")
print ("resultado teórico: ", 39/7)
```



O código retorna como output:


```python
Para facilitar o código, chamamos o estado 1 de 0, o 2 de 1, o 3 de 2, o 4 de 3 e o 5 de 4. Assim, o novo nome do estado coincide com seu índice na matriz. Dessa forma, usando a nova nomenclatura, temos:


Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.
resultado teórico: 8.7
```



O resultado teórico é 8.751. E, numericamente, o valor encontrado foi:  8.751



----

### Questão 3



Para simular os passos do Martingal, fizemos:



```python

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

```



A imagem retornada é:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-3-novo-martingal.png)



----

### Questão 4



```python
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

print (Nt, Tn)

plt.step(Tn, Nt + [Nt[-1]], where='post')

s=0
for i in range(100000):
    _, _, _, It = questao4_1()
    s += It

print (s/100000)

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

print (Nt, Tn)

plt.step(Tn, Nt + [Nt[-1]], where='post')

s=0
for i in range(100000):
    It, _, _, = questao4_2()
    s += It
print (s/100000)

plt.show()
```



A imagem retornada é:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-4.png)



----

### Questão 5



Seguindo a seção 9.5 das notas de aula e com a ajuda da Victória e do Hugo, colegas do curso, obtivemos:



```python
import numpy as np
import matplotlib.pyplot as plt

def kernel_f(x,xl,sigma,l):
    return sigma**2*np.exp(-5.*(1/l**2)*(x-xl)**2)

def K(X,Xl,sigma,l):
    n, m = X.shape[0],Xl.shape[0]
    K = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] = kernel_f(X[i], Xl[j], sigma, l)
    return K

def exercicio5(X, X_new, Y, sigma):
    KXX = K(X,X,1,1)
    KXnX = K(X_new, X, 1, 1)
    KXXn = K(X,X_new, 1, 1)
    KXnXn = K(X_new, X_new, 1, 1)

    aux = KXnX.dot(np.linalg.inv(KXX+sigma**2*np.identity(KXX.shape[0])))

    mean = aux.dot(Y)
    covariance = KXnXn + sigma**2*np.identity(KXnXn.shape[0]) - aux.dot(KXXn)

    return (mean,covariance)

n, m = 9, 1000

X = np.linspace(-5,5,n)
Y = np.sin(X)+np.random.normal(loc = 0, scale = 0.001, size=n)
Xn = np.linspace(-5,5,m)

mu, cov = exercicio5(X, Xn, Y, 0.01)

stdv = np.sqrt(np.diag(cov))

plt.plot(X,Y, 'bs', ms=8, label="Pontos iniciais")
plt.plot(Xn,mu,label="Média")
plt.gca().fill_between(Xn, mu-2*stdv,mu+2*stdv,color="#dddddd",label="intervalo de confiança")
plt.plot(Xn,mu,"y--",lw=2,label="função aproximada")
plt.axis([-5,5,-3,3])
plt.show()

```



O código retorna:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-5.png)



----

### Questão 6

O código é



```python
import matplotlib.pyplot as plt
import numpy as np

def brownian_motion(M, N, T): #M é o número de caminhos, n é o número de incrementos, T é o tempo final
    h = T/N
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


brownian_motion(1000, 100, 1)

```



Imagem que retorna:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-6.png)



----



### Questão 7



O código é:

```python
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

```



Imagem que retorna:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-7.png)