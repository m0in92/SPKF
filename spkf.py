import numpy as np


def f_func(x_k, u_k, w_k):
    return np.sqrt(5+x_k) + w_k

def h_func(x_k, u_k, v_k):
    return x_k**3 + v_k


class SPKF:
    def __init__(self, Nx, NXa, Ny, SigmaX, SigmaW, SigmaV):
        """
        Constructor for the SPKF class.
        :param Nx: number of state variables.
        :param NXa: number of state variables in the augmented state vector.
        :param Ny: number of outputs.
        :param SigmaX:
        :param SigmaW:
        :param SigmaV:
        """
        self.Nx = Nx
        self.Nxa = NXa
        self.L = self.Nxa
        self.Ny = Ny
        self.SigmaX = SigmaX
        self.SigmaW = SigmaW
        self.SigmaV = SigmaV
        self.p = 2 * self.L

    def state_estimate_time_update(self):
        pass

    def covariance_prediction(self):
        pass

    def output_estimate(self):
        pass

    def estimator_gain_matrix(self):
        pass

    def state_estimate_measurement_update(self):
        pass

    def error_covariance_measurement_update(self):
        pass