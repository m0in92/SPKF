import numpy as np
import matplotlib.pyplot as plt


## Initialization
import scipy.linalg

max_iter = 500
# State-space matrix
A = 1
B = -1e-4
C = 0.7
D = -0.01
# Variances
Sigma_w = 1e-5
Sigma_v = 0.1
# Initial conditions
x_true = 0.5
x_hat = 0.5
Sigma_X = 0
u_0 = 1

## Input generation
u = np.random.randn(max_iter) #  generates input noise
u[0]=u_0 # match the initial condition
w = scipy.linalg.cholesky(Sigma_w)[0] * np.random.randn(max_iter) # generate process noise
v = scipy.linalg.cholesky(Sigma_v)[0] * np.random.rand(max_iter) # generate sensor noise

# vectors for storage
x_true_vec = np.zeros(max_iter)
x_true_vec[0] = x_true
y_true_vec = np.zeros(max_iter)
y_true_vec[0] = x_true
y_hat_vec = np.zeros(max_iter)
y_hat_vec[0] = x_true
x_hat_vec = np.zeros(max_iter)
x_hat_vec[0] = x_hat
Sigma_X_vec = np.zeros(max_iter)
Sigma_X_vec[0] = Sigma_X

for iter in range(1,max_iter):
    # Step 1: state-space prediction
    x_hat = A*x_hat + B*u[iter-1]

    #Step 2: Error-covariance
    Sigma_X = A*Sigma_X*A + Sigma_w

    #Step 3: y estimate
    x_true_vec[iter] = A*x_true + B*u[iter] + w[iter]
    y_true = C*x_true_vec[iter] + D*u[iter] + v[iter]
    yhat = C*x_hat + D*u[iter]

    #Step 4: Gain matrix
    Sigma_y = C*Sigma_X*C + Sigma_v
    L = Sigma_X * C / Sigma_y

    # Step 5: state estimate
    xhat = x_hat + L*(y_true-yhat)

    # Step 6: Error-covariance update
    Sigma_X = Sigma_X - L*Sigma_y*L

    # Store calculated values
    y_true_vec[iter] = y_true
    y_hat_vec[iter] = yhat
    x_hat_vec[iter] = x_hat
    Sigma_X_vec[iter] = Sigma_X

plt.plot(y_true_vec)
plt.plot(y_hat_vec)
plt.plot(Sigma_X_vec)
plt.show()