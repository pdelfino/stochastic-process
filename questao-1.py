import numpy as np

P = np.array([[1/3,  0 , 2/3,  0 ,  0 ],
              [1/4, 1/2, 1/4,  0 ,  0 ],
              [1/2,  0 , 1/2,  0 ,  0 ],
              [ 0 ,  0 ,  0 ,  0 ,  1 ],
              [ 0 ,  0 ,  0 , 2/3, 1/3]])

#print (np.matmul(P,P))

'''for i in range(1,10):
    P = np.matmul(P,P)
''' 
