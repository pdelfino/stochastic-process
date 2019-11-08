# Stochastic Processing   
---

 + Programming assignments for the course "Processos Estocásticos" (Stochastic Process) at the Escola de Mamemática Aplicada (EMAp). 

 + Students: 
   - Pedro Delfino; 
   - Bruna Fistarol; and,
- Danillo Fiorenza.
   
 + Professor: PhD Yuri Saporito

 + Date: 2019.2.

 + Tech Stack: Python 3.6.8.  

    + **Atenção,** a específicação acima é especialmente importante para a função random.choices do pacote random. 

 + Exercises: 
   
   - see [problem set pdf](https://github.com/pdelfino/stochastic-process/blob/master/problem-set.pdf)
   
 + Support Material:

    +  [lecture notes](https://drive.google.com/file/d/0BwDJjYFvJgwNZFk1dmFKeExKblU/view) written by Professor Yuri; 
    + Introduction To Stochastic Process With R.

   

   ### Questão 1

   ----

   A imagem abaixo ilustra bem as probabilidades de transição. Além disso, é possível perceber que os estados 2 é o único transiente. Todos os outros são recorrentes.  Outra característica evidente é separação em dois grafos, de modo que os estados 1, 2 e 3 não se comunicam com os estados 4 e 5.                                                               : 

   ![](/home/pedro/Pictures/Screenshot from 2019-11-08 16-43-16.png)

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

```





O código retorna como output:


```python
Para facilitar o código, chamamos o estado 1 de 0, o 2 de 1, o 3 de 2, o 4 de 3 e o 5 de 4. Assim, o novo nome do estado coincide com seu índice na matriz. Dessa forma, usando a nova nomenclatura, temos:


Como é possível ver, o resultado analítico e o experimento de simulação computacional são convergentes.
resultado teórico:  5.571428571428571
```



