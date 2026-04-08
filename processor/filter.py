import torch
import math

class OneEuroFilter:
    """1 Euro Filter for PyTorch Tensors. Applies low-pass filter to smooth signal."""
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = torch.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        t_e_tensor = torch.tensor(t_e, device=x.device, dtype=x.dtype)
        
        if t_e <= 0:
            return x
            
        a_d = self.smoothing_factor(t_e_tensor, self.dcutoff)
        dx = (x - self.x_prev) / t_e_tensor
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.mincutoff + self.beta * torch.abs(dx_hat)
        a = self.smoothing_factor(t_e_tensor, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
