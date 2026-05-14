"""
File: system.py

Program that defines a basic quantum wave system.

Programmer: Neil Ghugare

Revision History:
    05/14/2025 Initial version with comments.

Notes: ---
"""

import numpy as np

from typing import Any

class QuantumSystem:
    """
    Defines a quantum system for use in our numerical animations.
    """

    def __init__(self, x_range: list, nx: int, dt: float, mass: float=1.0) -> None:
        """
        Initializes a new system.

        Arguments (required)
        1. x_range - The x_range being simulated on.
        2. nx - The number of x "cells".
        3. dt - The time step being used.

        Arguments (optional)
        1. mass - The mass of the particle. Defaults to 1.0.

        Returns
        Nothing.
        """

        # Define the x range, dx, dt, and mass of the system.
        self.x = np.linspace(*x_range, nx)
        self.dx = self.x[1]-self.x[0]
        self.dt = dt
        self.mass = mass

        # Momentum space setup for the Split-Step Fourier Method (SSFM).
        self.k = 2*np.pi * np.fft.fftfreq(nx, d=self.dx)                        # The wavenumber k.
        self.kinetic_oper = np.exp(-1j* (self.k**2)*self.dt / (2*self.mass))    # The kinetic operator.
    
    def get_evolution_operator(self, V: Any) -> Any:
        """
        Gets the evolution operator for this system.

        Arguments (required)
        1. V - The potential.

        Arguments (optional)
        None

        Returns
        The evolution operator exp(-i*V*dt). 
        """
        return np.exp(-1j*V*self.dt)