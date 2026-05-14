import numpy as np

from abc import ABC, abstractmethod

class Potential(ABC):
    @abstractmethod
    def get_V(self, x: np.ndarray) -> np.ndarray:
        """
        """
        pass

    @abstractmethod
    def get_manim_points(self, axes, x_range):
        """"""
        pass

class RectangularBarrier(Potential):
    def __init__(self, height=15, left=2, right=3):
        self.height = height
        self.left = left
        self.right = right

    def get_V(self, x):
        return np.where((x > self.left) & (x < self.right), self.height, 0)

    def get_manim_points(self, axes, x_range):
        h_vis = self.height * 0.2
        coords = [
            [x_range[0], 0, 0], [self.left, 0, 0],
            [self.left, h_vis, 0], [self.right, h_vis, 0],
            [self.right, 0, 0], [x_range[1], 0, 0]
        ]
        return [axes.c2p(*c) for c in coords]

class SemiCircularBarrier(Potential):
    def __init__(self, height=15, center=2.5, radius=0.5):
        self.height = height
        self.center = center
        self.radius = radius

    def get_V(self, x):
        dist = (x - self.center) / self.radius
        # Ensure the argument for sqrt is at least 0 to avoid warnings
        inner_term = np.maximum(0, 1 - dist**2)
        
        # Now sqrt only sees non-negative numbers
        v = np.where(np.abs(dist) <= 1, self.height * np.sqrt(inner_term), 0)
        return v

    def get_manim_points(self, axes, x_range):
        # For a curve, we sample points more densely
        x_vals = np.linspace(self.center - self.radius, self.center + self.radius, 40)
        v_vals = self.height * 0.2 * np.sqrt(1 - ((x_vals - self.center) / self.radius)**2)
        
        points = [axes.c2p(x_range[0], 0), axes.c2p(self.center - self.radius, 0)]
        points += [axes.c2p(x, y) for x, y in zip(x_vals, v_vals)]
        points += [axes.c2p(self.center + self.radius, 0), axes.c2p(x_range[1], 0)]
        return points