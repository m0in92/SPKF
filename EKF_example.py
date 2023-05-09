import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


## Initialization
Sigma_W = 1
Sigma_V = 2
max_iter = 40
x_true = 2 + np.random.normal() # Initialize true system initial state
x_hat = 2 # intialize state estimate
Sigma_X = 1 # intialize Kalman filter covariance
u = 0 # initial driving input

# Initialize vectors for storage and update
x_true_vec = np.zeros(max_iter+1)
x_true_vec[0] = x_true # Set the first entry of the vector to the initial state value
x_hat_vec = np.zeros(max_iter) # State estimates vector
# x_hat_vec[0] = x_hat
Sigma_X_vec = np.zeros(max_iter)

for iter in range(max_iter):
    # Step 1a: state prediction
    A_hat = 1/(2*np.sqrt(5+x_hat))
    B_hat = 1
    x_hat = np.sqrt(5+x_hat)

    # Step 1b: Error-covariance time update
    Sigma_X = A_hat*Sigma_X*np.transpose(A_hat) + B_hat*Sigma_W*np.transpose(B_hat)

    # Step 1c: output estimate
    v = scipy.linalg.cholesky(Sigma_V)*np.random.normal()
    w = np.transpose(scipy.linalg.cholesky(Sigma_W)) * np.random.normal()
    y_true = x_true**3 + v
    x_true = np.sqrt(5+x_true) + w

    C_hat = 3*(x_hat**2)
    D_hat = 1
    y_hat = x_hat**3

    # Step 2a: Gain Factor
    Sigma_Y = C_hat*Sigma_X*np.transpose(C_hat) + D_hat*Sigma_V*np.transpose(D_hat)
    L = Sigma_X*np.transpose(C_hat)/Sigma_Y

    # Step 2b: State estimate
    x_hat = x_hat + L*(y_true - y_hat)
    x_hat = max(-5, x_hat) # Avoids the square root of a negative number

    # Step 2c: Error-covariance measurement update
    Sigma_X = Sigma_X - L*Sigma_Y*np.transpose(L)
    # _,s,Vh = scipy.linalg.svd(Sigma_X)
    # HH = Vh*s*np.transpose(Vh)

    # Update the vectors
    x_hat_vec[iter] = x_hat
    x_true_vec[iter+1] = x_true
    Sigma_X_vec[iter] = Sigma_X

plt.plot(range(0,max_iter+1), x_true_vec, label="Truth")
plt.plot(range(0,max_iter), x_hat_vec, label="EFK est.")
plt.plot(range(0,max_iter), x_hat_vec + Sigma_X_vec, "g:",label="Bounds")
plt.plot(range(0,max_iter), x_hat_vec - Sigma_X_vec, "g:")
plt.xlabel('Iteration')
plt.ylabel('State')
plt.legend()
plt.show()