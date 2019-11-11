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

    + **Atenção,** a especificação acima é especialmente importante para a função random.choices do pacote random. 

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

A demonstração do valor teórico está na imagem abaixo. Infelizmente, não foi possível, no tempo disponível, usar o LaTeX dentro desse Markdown:

![](https://github.com/pdelfino/stochastic-process/blob/master/teorica-1.jpg)

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
print ("resultado teórico: ", 133/15)
```



O código retorna como output:


```python
Para facilitar o código, chamamos o estado 1 de 0, o 2 de 1, o 3 de 2, o 4 de 3 e o 5 de 4. Assim, o novo nome do estado coincide com seu índice na matriz. Dessa forma, usando a nova nomenclatura, temos:


Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.
resultado teórico: 8.86
```



O resultado teórico é 8.86. E, numericamente, o valor encontrado foi:  8.683366733466935



A demonstração do valor teórico está na imagem abaixo. Infelizmente, não foi possível, no tempo disponível, usar o LaTeX dentro desse Markdown:

![](https://github.com/pdelfino/stochastic-process/blob/master/teorica-2.jpg)



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

Nesse caso, cada salto é uma chegada.

O output gerado é:

```python
s/100000 no tópico (i):  12.503331366866945
s/100000 no tópico (ii): 12.508935816992528
```

O resultado teórico é uma integral simples que resulta em `25/2`,isto é, exatamente 12.5.

Portanto, o resultado teórico e empírico-numérico-estocástico é o mesmo.

----

### Questão 5



Seguindo a seção 6.3  das notas de aula e com a ajuda da Victória e do Hugo, colegas do curso, conseguimos os resultados abaixo.

Cabe ainda, fazer algumas explicações: 

- o vetor X tem poucos pontos
- o que está em jogo é uma distribuição Bayesiana e queremos descobrir a distribuição a posteriori.
- Queremos a distribuição de Y* no intervalo fornecido.
- Era para fazer X* de -pi até +pi, acabamos fazendo de -5 a +5 pois fomo como a Victória explicou
- Queremos a distribuição de Y*, dado X, X* e Y
- Nas notas temos os valores da média e da covariance
- Y* é uma normal multivariada



```python
import numpy as np
import matplotlib.pyplot as plt

#xl é o x linha
# exemplo 6.2.2 da apostila
def nucleo(x,xl,sigma,l):
    
    return sigma**2*np.exp(-5.*(1/l**2)*(x-xl)**2)


# calculando a matriz do Kzão
def matrix_K(X,Xl,sigma,l):
    
    n = X.shape[0]
    
    m = Xl.shape[0]
    
    K = np.zeros((n,m))
    
    for i in range(n):
        
        for j in range(m):
            
            K[i,j] = nucleo(X[i], Xl[j], sigma, l)
    
    return K

#retorna a média e a covarianceariância, pag 95 da apostila
def media_covariancear(X, X_new, Y, sigma):
    KXX = matrix_K(X,X,1,1)
    KXnX = matrix_K(X_new, X, 1, 1)
    KXXn = matrix_K(X,X_new, 1, 1)
    KXnXn = matrix_K(X_new, X_new, 1, 1)

    aux = KXnX.dot(np.linalg.inv(KXX+sigma**2*np.identity(KXX.shape[0])))

    media = aux.dot(Y)
    covariance = KXnXn + sigma**2*np.identity(KXnXn.shape[0]) - aux.dot(KXXn)

    return (media,covariance)

n = 9
m = 1000

X = np.linspace(-5,5,n)
Y = np.sin(X)+np.random.normal(loc = 0, scale = 0.001, size=n)
Xn = np.linspace(-5,5,m)

media = media_covariancear(X, Xn, Y, 0.01)[0]
covariance = media_covariancear(X, Xn, Y, 0.01)[1]

stdv = np.sqrt(np.diag(covariance))

plt.plot(X,Y, 'bs', ms=8, label="Initial points")

plt.plot(Xn,media,label="Mean")

plt.gca().fill_between(Xn, media-2*stdv,media+2*stdv,color="#dddddd",label="confidence interval")

plt.plot(Xn,media,"y--",lw=2,label="approximate function")

plt.axis([-5,5,-3,3])

plt.legend()

plt.show()


```



O código retorna:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-5.png)

O Y é o seno de x mais o ruído (epsolon). A área de cinza expressa o intervalo de confiança.

A linha de amarela é a função seno

A linha em verde pontilhada é a média

O ponto de azul é o observado

O trace

----

### Questão 6

O código é:

```python
import matplotlib.pyplot as plt
import numpy as np

def brownian_motion(M, N, T): 
    # M é o número de caminhos, 
    # n é o número de incrementos, 
    # T é o tempo final
    
    # h é o tamanho do intervalo
    h = T/N
    
	# z é a matriz de tamanho Mx(N-1) de N(0,h)
    Z = np.sqrt(h) * np.random.normal(size = (M, N-1))
    
    # B é o vetor de zeros MxN
    B = np.zeros(shape = (M, N))
    
	#B[:,1:] é a soma acumulada nas linhas de Z
    B[:, 1:] = np.cumsum(Z, axis = 1)
    
    # max de cada linha de B
    m = np.max(B, axis = 1)
    
    plt.plot(np.linspace(0, 1, N), B[:10, :].T)
    plt.show()
    x = np.linspace(0, 3.5, 100)
    y = ((2/(np.pi))**(1/2))*(np.exp((-x**2)/(2)))
    plt.plot(x, y)
    plt.hist(m, normed = True)


brownian_motion(1000, 100, 1)

```



A imagem que o código retorna reflete a forma gráfica  típica de um movimento browniano. Apesar de contínua, a função gerada não é diferenciável em vários pontos por não ser "smooth", haja vista as várias "quinas":

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-6.png)

Cada uma das curvas de diferentes cores simboliza um caminho distinto. O eixo X simboliza o tempo e o eixo Y, por sua vez, representa a posição no tempo.

Outro comentário interessante que pode ser feito é sobre a grande probabilidade **(quase certamente)** de que os caminhos ficam entre `y=sqrt(x)` e `y=-sqrt(x`.

O histograma abaixo, por sua vez, indica o ponto máximo alcançando por cada caminho:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-6-hist.png)

Um comentário interessante que pode ser feito é a comparação entre a desindade exata da distribuição normal padronizada (em azul) e a distribuição encontrada empiricamente (em laranja) para o ponto máximo atingido.

----



### Questão 7

A fórmula de Black-Scholes está inserida no contexto de opções. Isto é, trata-se de um contrato financeiro em que uma das partes compra a opção de comprar um ativo (ação, commoditie, instrumento ou título de dívida). Há de ser ressaltado que a opção não vincula o comprador, isto é, ele não é obrigado a exercê-la.

O código é:

```python
import numpy as np
import matplotlib.pyplot as plt

def black_scholes(n): #n é o número de valores S gerados

    # um vetor com valores dentro da distribuição normal padronizada
    W = np.random.normal(size = n)

    # S (Current Stock) é o preço atual da opção calculado em cima dos valores de W
    S = 100*np.exp(-0.055 + W*0.04)

    # Preço fixado (Striking price) da opção (pode ser de compra ou de venda)
    K = np.linspace(80, 120, 9)
	
	# C é o Preço da Opção (Call Option Price)
    C = np.zeros(shape = 9)
    for k in range(9):
        C[k] = np.mean(np.exp(-0.05) * np.maximum(S - K[k], 0.0))
    
    plt.plot(K, C)
    plt.show()

print (black_scholes(100))

```

Apenas para esclarecimento, é importante diferenciar a call option price (C), o current price (S) e o strike price (K). Nesse sentido, vamos usar um exemplo da ação da Petrobrás.

Suponhamos que a ação da Petrobrás valha  10 reais **hoje** (11 de Novembro de 2019). Esse é o current price, isto é, preço atual. Suponhamos ainda que o investidor *Yuri Lobo de Faria Lima* deseja comprar uma opção sobre esta ação.

Assim, suponhamos que o preço da opção no mercado seja 50 centavos. No contrato financeiro, Yuri tem o direito de exercer essa compra até 31 de Dezembro de 2019 e de comprar a ação por 12 reais. Nesse caso, 50 centavos é o Call option price e 12 reais é o Striking Price.

Depois de uma revolução e uma verdadeira faxina no país, vários corruptos voltam para a cadeia (#lulaPreso) e a ação da Petrobrás passou a valer 100 reais no final de Novembro. Yuri, muito esperto, decide exercer sua opção de compra. Dessa forma, o nosso investidor exerce sua opção de compra, adquire a ação da Petrobrás por 12 reais e, no instante seguinte, a vende no mercado por 100 reais. Com isso, faz um belo retorno de 87.5 reais.



Imagem que retorna:

![](https://github.com/pdelfino/stochastic-process/blob/master/questao-7.png)