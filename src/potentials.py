"""
File: system.py

Barrier potentials and general potentials for the quantum system.

Programmer: Neil Ghugare

Revision History:
    05/14/2025 Initial version with comments.

Notes:
Implement a custom potential by inheriting the Potential abstract base class.
"""

import numpy as np

from abc import ABC, abstractmethod
from typing import Any

class Potential(ABC):
    """
    An abstract base class representing a Potential V. 
    """

    @abstractmethod
    def get_V(self, x: np.ndarray) -> np.ndarray:
        """
        An abstract method that gets the potential values V(x).

        Arguments (required)
        1. x - The x-range to get V(x) for.

        Arguments (optional)
        None

        Returns
        Should return V(x), but this abstract method does nothing.
        """
        pass

    @abstractmethod
    def get_manim_points(self, axes: Any, x_range: list | tuple) -> list:
        """
        An abstract method that gets the Manim points for the potential.

        Arguments (required)
        1. axes - The Manim axes.
        2. x_range - The x-range for the Manim animation.

        Arguments (optional)
        None

        Returns
        Should return a list of Manim points, but this abstract method does nothing.
        """
        pass

class RectangularBarrier(Potential):
    """
    An implementation of the Potential class for a rectangular barrier.
    """

    def __init__(self, height: float=15, left: int=2, right: int=3) -> None:
        """
        Initializes a rectangular barrier potential.

        Arguments (required)
        None

        Arguments (optional)
        1. height - The height of the barrier. Defaults to 15.
        2. left - The left x-coordinate of the barrier. Defaults to 2.
        3. right - The right y-coordinate of the barrier. Defaults to 3.

        Returns
        Nothing.
        """

        # Set the field values.
        self.height = height
        self.left = left
        self.right = right

    def get_V(self, x: np.ndarray) -> np.ndarray:
        """
        Gets the potential V(x) for the rectangular barrier.

        Arguments (required)
        1. x - The x-range to get V(x) for.

        Arguments (optional)
        None

        Returns
        The values V(x).
        """

        # Return the barrier values based on some array comprehension.
        return np.where((x > self.left) & (x < self.right), self.height, 0)

    def get_manim_points(self, axes: Any, x_range: list | tuple) -> list:
        """
        Gets the Manim points for the rectangular barrier.

        Arguments (required)
        1. axes - The Manim axes.
        2. x_range - The x-range for the Manim animation.

        Arguments (optional)
        None

        Returns
        A list of Manim points for the barrier via c2p. 
        """

        # Scale the height for visualization
        h_vis = self.height * 0.2

        # Set manual coordinates so the vertical lines are drawn properly in Manim.
        coords = [
            [x_range[0], 0, 0], [self.left, 0, 0],
            [self.left, h_vis, 0], [self.right, h_vis, 0],
            [self.right, 0, 0], [x_range[1], 0, 0]
        ]

        # Return the coordinates to points.
        return [axes.c2p(*c) for c in coords]

class SemiCircularBarrier(Potential):
    """
    An implementation of the Potential class for a semi-circular/elliptical barrier.
    """

    def __init__(self, height=15, center=2.5, radius=0.5):
        """
        Initializes a rectangular barrier potential.

        Arguments (required)
        None

        Arguments (optional)
        1. height - The height of the barrier. Defaults to 15.
        2. center - The center x-value of the barrier. Defaults to 2.5.
        3. radius - The radius of the barrier. Defaults to 0.5.

        Returns
        Nothing.
        """

        # Set the field values.
        self.height = height
        self.center = center
        self.radius = radius

    def get_V(self, x: np.ndarray) -> np.ndarray:
        """
        Gets the potential V(x) for the semi-circular barrier.

        Arguments (required)
        1. x - The x-range to get V(x) for.

        Arguments (optional)
        None

        Returns
        The values V(x).
        """

        # Get the distances from the points to the center, scaled by the radius.
        dist = (x - self.center) / self.radius
        
        # Ensure the argument for sqrt is at least 0 to avoid warnings.
        inner_term = np.maximum(0, 1 - dist**2)
        
        # Now we sqrt only sees non-negative numbers and return the potential value.
        v = np.where(np.abs(dist) <= 1, self.height * np.sqrt(inner_term), 0)
        return v

    def get_manim_points(self, axes: Any, x_range: list | tuple) -> list:
        """
        Gets the Manim points for the semi-circular barrier.

        Arguments (required)
        1. axes - The Manim axes.
        2. x_range - The x-range for the Manim animation.

        Arguments (optional)
        None

        Returns
        A list of Manim points for the barrier via c2p. 
        """

        # For a curve, we sample points more densely.
        # We then get the V values.
        x_vals = np.linspace(self.center - self.radius, self.center + self.radius, 40)
        v_vals = self.height * 0.2 * np.sqrt(1 - ((x_vals - self.center) / self.radius)**2)
        
        # Get the points using c2p and return them.
        points = [axes.c2p(x_range[0], 0), axes.c2p(self.center - self.radius, 0)]
        points += [axes.c2p(x, y) for x, y in zip(x_vals, v_vals)]
        points += [axes.c2p(self.center + self.radius, 0), axes.c2p(x_range[1], 0)]
        return points