import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EKF:
    """
    EKF for Thevenin 1RC only! Under construction
    """
    def __int__(self, C_nominal, R0, R1, C1, OCV_class, Sigma_W, Sigma_V, SOC_0, i1_0, t_app, i_app, V_actual):
        self.nx = 2
        self.C_nominal = C_nominal
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.OCV = OCV_class
        self.Sigma_W = Sigma_W
        self.Sigma_V = Sigma_V
        self.t_app = t_app
        self.z_hat = SOC_0  # intialize state estimate
        self.i1_hat = i1_0
        self.Sigma_X = np.zeros([nx, nx])  # intialize Kalman filter covariance
        self.Sigma_X[0, 0] = 0
        self.Sigma_X[1, 0] = 0
        self.x_hat = np.zeros([nx, 1])
        self.x_hat[0, 0] = self.z_hat
        self.x_hat[1, 0] = self.i1_hat

    def solve(self):
        max_iter = len(self.t_app)
        # Initialize vectors for storage and update
        z_hat_vec = np.zeros(max_iter)  # State estimates vector
        z_hat_vec[0] = z_hat
        Sigma_X_vec = np.zeros(max_iter)
        y_hat_vec = np.zeros(max_iter)

        for iter in range(1, max_iter):
            delta_t = self.t_app[iter] - self.t_app[iter - 1]

            # Step 1a: state prediction
            A = A_hat = np.zeros([self.nx, self.nx])
            A[0, 0] = 1
            A[1, 1] = np.exp(-delta_t / (self.R1 * self.C1))
            A_hat[0, 0] = 1
            A_hat[0, 1] = -delta_t / self.C_nominal
            A_hat[1, 1] = np.exp(-delta_t / (self.R1 * self.C1))
            B = B_hat = np.zeros([self.nx, 1])
            B[0, 0] = -delta_t / self.C_nominal
            B[1, 0] = 1 - np.exp(-delta_t / (self.R1 * self.C1))
            B_hat[0, 0] = -delta_t / self.C_nominal
            B_hat[1, 0] = 1 - np.exp(-delta_t / (self.R1 * self.C1))

            x_hat = np.dot(A, x_hat) + np.dot(B, i_app[iter - 1])

            # Step 1b: Error-covariance time update
            self.Sigma_X = np.dot(np.dot(A_hat, self.Sigma_X), np.transpose(A_hat)) + np.dot(np.dot(B_hat, self.Sigma_W),
                                                                                   np.transpose(B_hat))

            # Step 1c: output estimate
            y_hat = self.OCV.OCV25_func(x_hat[0, 0]) - self.R1 * x_hat[1, 0] - self.R0 * self.i_app[iter]

            # Step 2a: Gain Factor
            C_hat = np.zeros([1, self.nx])
            C_hat[0, 0] = self.OCV.OCV25_diff_func(x_hat[0, 0])
            C_hat[0, 1] = -self.R1
            D_hat = 1
            Sigma_Y = np.dot(np.dot(C_hat, self.Sigma_X), np.transpose(self.C_hat)) + np.dot(np.dot(self.D_hat, self.Sigma_V),
                                                                                   np.transpose(D_hat))
            L = np.dot(self.Sigma_X, np.transpose(C_hat)) / self.Sigma_Y

            # Step 2b: State estimate
            x_hat = x_hat + np.dot(L, (self.V_actual[iter] - y_hat))

            # Step 2c: Error-covariance measurement update
            Sigma_X = Sigma_X - np.dot(np.dot(L, Sigma_Y), np.transpose(L))
            # _,s,Vh = scipy.linalg.svd(Sigma_X)
            # HH = Vh*s*np.transpose(Vh)

            # Update the storage vectors
            z_hat_vec[iter] = x_hat[0, 0]
            Sigma_X_vec[iter] = Sigma_X[0, 0]
            y_hat_vec[iter] = y_hat

        return z_hat_vec, Sigma_X_vec, y_hat_vec


# EKF Parameters
#---------------------------------------------------------------------------------
nx = 2
Sigma_W = 2e-6
Sigma_V = 2e-8
max_iter = len(t_app)
z_hat = SOC_0 # intialize state estimate
i1_hat = 0
Sigma_X = np.zeros([nx,nx]) # intialize Kalman filter covariance
Sigma_X[0,0] = 0
Sigma_X[1,0] = 0
x_hat = np.zeros([nx,1])
x_hat[0,0] = z_hat
x_hat[1,0] = i1_hat

# Initialize vectors for storage and update
z_hat_vec = np.zeros(max_iter) # State estimates vector
z_hat_vec[0] = z_hat
Sigma_X_vec = np.zeros(max_iter)
y_hat_vec = np.zeros(max_iter)

for iter in range(1,max_iter):
    delta_t = t_app[iter] - t_app[iter - 1]

    # Step 1a: state prediction
    A = A_hat = np.zeros([nx,nx])
    A[0,0] = 1
    A[1,1] = np.exp(-delta_t/(R1*C1))
    A_hat[0,0] = 1
    A_hat[0,1] = -delta_t/C_nominal
    A_hat[1,1] = np.exp(-delta_t/(R1*C1))
    B = B_hat = np.zeros([nx,1])
    B[0,0] = -delta_t/C_nominal
    B[1, 0] = 1 - np.exp(-delta_t / (R1 * C1))
    B_hat[0,0] = -delta_t/C_nominal
    B_hat[1,0] = 1 - np.exp(-delta_t/(R1*C1))

    x_hat = np.dot(A,x_hat) + np.dot(B,i_app[iter-1])

    # Step 1b: Error-covariance time update
    Sigma_X = np.dot(np.dot(A_hat, Sigma_X),np.transpose(A_hat)) + np.dot(np.dot(B_hat, Sigma_W), np.transpose(B_hat))

    # Step 1c: output estimate
    y_hat = OCV.OCV25_func(x_hat[0,0]) - R1*x_hat[1,0] - R0*i_app[iter]

    # Step 2a: Gain Factor
    C_hat = np.zeros([1,nx])
    C_hat[0,0] = OCV.OCV25_diff_func(x_hat[0,0])
    C_hat[0,1] = -R1
    D_hat = 1
    Sigma_Y = np.dot(np.dot(C_hat, Sigma_X), np.transpose(C_hat)) + np.dot(np.dot(D_hat, Sigma_V), np.transpose(D_hat))
    L = np.dot(Sigma_X, np.transpose(C_hat)) / Sigma_Y

    # Step 2b: State estimate
    x_hat = x_hat + np.dot(L, (V_actual[iter] - y_hat))

    # Step 2c: Error-covariance measurement update
    Sigma_X = Sigma_X - np.dot(np.dot(L, Sigma_Y), np.transpose(L))
    # _,s,Vh = scipy.linalg.svd(Sigma_X)
    # HH = Vh*s*np.transpose(Vh)

    # Update the storage vectors
    z_hat_vec[iter] = x_hat[0,0]
    Sigma_X_vec[iter] = Sigma_X[0,0]
    y_hat_vec[iter] = y_hat

# Plots
### Plot for time-voltage
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(t_app, V_actual, color = "black",label="Actual")
ax1.plot(t_app[1:], y_hat_vec[1:], label="ECM", linewidth=1.5, color = "tab:blue")
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cell Terminal Voltage [V]')
ax1.set_title('Thevenin-1RC with EFK')
ax1.legend()

ax2 = fig.add_subplot(2,1,2)
ax2.plot(t_app[1:], V_actual[1:] -  y_hat_vec[1:])
# ax2.plot(t_app[1:], Sigma_X_vec[1:])
# ax2.plot(t_app[1:], -Sigma_X_vec[1:])
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Difference [V]')
ax2.set_ylim(-0.11,0.11)

plt.tight_layout()
plt.show()

# ## Plot for SOC
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(t_app, z_hat_vec, label="ECM", linewidth=1.5, color = "tab:blue")
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Cell Terminal Voltage [V]')
# ax1.set_title('Thevenin-1RC')
# ax1.legend()
#
# plt.tight_layout()
# plt.show()