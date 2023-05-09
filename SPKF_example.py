"""
This is an example of SPKF from Plett's book "Battery Management System. vol 2".

x_{k+1} = f(x_k, u_k, w_k) = sqrt(5+x_k) + w_k
y_k = h(x_k, u_k, v_k) = x_3 ** 3 + v_k

"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


# Define the size of variables in the model
Nx = 1
Nxa = 3 #Augmented state has x, w(k), and v(k)
Ny = 1

# SPKF constants
h = np.sqrt(3)
alpha1 = (h*h-Nxa)/(h*h)
alpha2 = 1/(2*h*h)
alpham = np.array(np.tile(alpha2, [2*Nxa,1]))
alpham = np.append(alpha1, alpham).reshape(-1,1)
alphac = alpham

# Initialize simulation variables
SigmaW = 1
SigmaV = 2
maxIter = 40
xtrue = 2 + np.random.normal()
xhat = 2
SigmaX = 1

# Initialize vectors for storage
xstore = np.zeros(maxIter+1)
xstore[0] = xtrue
xhatstore = np.zeros(maxIter)
SigmaXstore = np.zeros(maxIter)

for i in range(maxIter):
    # Step 1a: state-estimate time update
    xhata = np.array([xhat,0,0]).reshape(-1,1)
    SigmaXa = np.diag([SigmaX, SigmaW, SigmaV])
    sSigmaXa = scipy.linalg.cholesky(SigmaXa, lower=True)

    X = np.tile(xhata, [1,2*Nxa+1]) + h * np.append(np.zeros((Nxa,1)), np.append(sSigmaXa, -sSigmaXa, axis=1), axis=1)
    Xx = np.sqrt(5+X[0,:]) + X[1,:]
    xhat = np.matmul(Xx, alpham)[0]

    # Step 1b: Covariance of prediction
    Xs = Xx - np.tile(xhat, [1, 2 * Nxa + 1])
    SigmaX = np.matmul(Xs, np.matmul(np.diag(alphac.flatten()), Xs.transpose()))[0,0]

    # Input signal
    w = (np.transpose(scipy.linalg.cholesky(SigmaW)) * np.random.normal())[0,0]
    v = (scipy.linalg.cholesky(SigmaV)*np.random.normal())[0,0]
    ytrue = xtrue ** 3 + v
    xtrue = np.sqrt(5 + xtrue) + w

    # Step 1c: Output estimate
    Y = Xx**3 + X[2,:]
    yhat = np.matmul(Y,alpham)

    # Step 2a: Estimator gain matrix
    Ys = Y - np.tile(yhat, [1,2*Nxa+1])
    SigmaXY = np.matmul(Xs, np.matmul(np.diag(alphac.flatten()), Ys.transpose()))
    SigmaY = np.matmul(Ys, np.matmul(np.diag(alphac.flatten()), Ys.transpose()))
    Lx = SigmaXY/SigmaY

    # Step 2b: State-estimate measurement update
    xhat = (xhat + Lx*(ytrue-yhat))[0,0]

    # Step 2c: Error co-variance measurement update
    SigmaX = (SigmaX - Lx*SigmaY*Lx.transpose())[0,0]

    # Data storage
    xhatstore[i] = xhat
    xstore[i + 1] = xtrue
    SigmaXstore[i] = SigmaX

plt.plot(range(0,maxIter+1), xstore, label="Truth")
plt.plot(range(0,maxIter), xhatstore, label="SPKF est.")
plt.plot(range(0,maxIter), xhatstore + SigmaXstore, "g:",label="Bounds")
plt.plot(range(0,maxIter), xhatstore - SigmaXstore, "g:")
plt.xlabel('Iteration')
plt.ylabel('State')
plt.legend()
plt.show()