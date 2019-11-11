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

