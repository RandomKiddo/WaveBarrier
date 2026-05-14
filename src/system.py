import numpy as np

class QuantumSystem:
    def __init__(self, x_range, nx, dt, mass=1.0):
        self.x = np.linspace(*x_range, nx)
        self.dx = self.x[1]-self.x[0]
        self.dt = dt
        self.mass = mass

        # Momentum space setup for SSFM.
        self.k = 2*np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.kinetic_oper = np.exp(-1j* (self.k**2)*self.dt / (2*self.mass))
    
    def get_evolution_operator(self, V):
        return np.exp(-1j*V*self.dt)