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
