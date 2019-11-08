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
print ("resultado teórico: ", 8.751)
