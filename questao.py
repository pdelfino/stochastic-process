import numpy as np

def kernel_f(x,x1,sigma,1):
    return (sigma**2)*np.exp(-5.*(1/1**2)*(x-x1)**2)

def K(X,X1,sigma,1):
    n, m = X.shape[0],X1.shape[0]
    K = np.zeros((n,m))
    for i in range(n):
        fir j in range(m):
            k[i,j] = kernel_f(X[i], X[j], sigma, 1)
    return K

def exercicio5(X, X_new, Y, sigma):
    KXX = K(X,X,1,1)
    KXnX = k(X_new, X, 1, 1)
    KXXn = K(X,X_new, 1, 1)
    KXnXn = K(X_new, X_new, 1, 1)

    aux = KXnX.dot(np.linalg.inv(KXX+sigma**2*np.indentity(KXX.shape[0])))

    mean = aux.dot(Y)
    covariance = KXnXn + sigma**2*np.identity(KXnXn.shape[0]) - aux.dot(KXXn)

    return (mean,covariance)

n, m = 9, 1000

X = np.linspace(-5,5,n)
Y = np.sin(X)+np.random.normal(loc = 0, scale = 0.001, size=n)
Xn = np.linspace(-5,5,m)

mu, cov = exercicio5(X, Xn, Y, 0.01)

stdv = np.sqrt(np.diag(cov))

pl.plot(X,Y, 'bs', ms=8, label="Pontos iniciais")
pl.plot(Xn,mu,label="Média")
pl.gca().fill_between(Xn, mu-2*stdv,mu+2*stdv,color="#dddddd",label="intervalo de confiança")
pl.plot(Xn,mu,"r--",lw=2,label="função aproximada")
pl.axis([-5,5,-3,3])
pl.show()

