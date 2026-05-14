"""
File: animation.py

Program that makes the Manim animations.

Programmer: Neil Ghugare

Revision History:
    05/14/2025 Initial version with comments.

Notes:
Use with "manim animation.py" with whatever Manim flags you need.
For example, "manim -qm -v WARNING -o out.mp4 animation.py".
"""

import numpy as np

from manim import *
from dataclasses import dataclass 
from system import QuantumSystem
from scipy.fft import fft, ifft
from potentials import *


# --- DATA CLASS ---

@dataclass(frozen=True)
class SystemParams:
    """
    A data class to hold system parameters for dynamic animations.
    """

    # --- Simulation Values ---
    nx: int = 2048                                          # The number of x cells.
    view_min: int = -10                                     # The minimum x-value that will be animated.
    view_max: int = 10                                      # The maximum x-value that will be animated.
    sim_min: int = -13                                      # Padded simulation minimum x for "sponging".
    sim_max: int = 13                                       # Padded simulation maximum x for "sponging".
    dt: float = 0.01                                        # The time step being used.

    # --- Physical Parameters ---
    mass: float = 1.0                                       # The mass of the particle.
    V_type: Potential = SemiCircularBarrier(radius=1.0)     # The potential barrier being used.

# --- ANIMATION SCENE ---

class QuantumAnimation(Scene):
    """
    Inherits the Manim Scene class to create the scene for this animation.
    """

    # The parameters for this animation.
    # This an be overridden via QuantumAnimation.params = SystemParams(**kwargs).
    params: SystemParams = SystemParams()

    def construct(self) -> None:
        """
        Constructs the animation scene.

        Arguments (required)
        None

        Arguments (optional)
        None

        Returns None
        """

        # To remove verbosity, so we don't see self.params everywhere.
        p = self.params

        # Solver uses the larger sim range than being visualized.
        # We instantiate the quantum system.
        sys = QuantumSystem((p.sim_min, p.sim_max), p.nx, p.dt, mass=p.mass)

        # Get the barrier we're using.
        barrier = p.V_type
        
        # --- 1. Physics Setup ---
        
        # The real potential value.
        v_real = barrier.get_V(sys.x)
        
        # We use "sponges", ghost cells to simulate further than view.
        # This allows the wave packet to travel off screen cleanly without periodic self-interference.
        # We place sponges strictly outside the view range on both the left and right sides.
        # They start at +/- 10.5 and ramp up toward the SIM edges +/- 13
        # todo generalize
        strength = 150  
        v_imag_l = np.where(sys.x < -10.5, strength * (sys.x + 10.5)**2, 0)
        v_imag_r = np.where(sys.x > 10.5, strength * (sys.x - 10.5)**2, 0)
        
        # The total V and V operator.
        v_total = v_real - 1j * (v_imag_l + v_imag_r)
        v_oper = np.exp(-1j * v_total * sys.dt)

        # --- 2. Initial Wavepacket ---

        # We place the initial wave as a Gaussian with pre-defined sigma, x0, and k0.
        # We can then calculate psi from this, making sure it is normalized.
        x0, sigma, k0 = -6, 0.6, 10
        psi = np.exp(-0.5 * ((sys.x - x0) / sigma)**2) * np.exp(1j * k0 * sys.x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * sys.dx)

        # --- 3. Visualization ---

        # Define the axes ranges based on the view window. 
        axes = Axes(x_range=[p.view_min, p.view_max, 1], y_range=[-2, 5, 1], 
                    axis_config={"include_tip": False})
        
        # Visualize the barrier itself using a VMobject, using get_manim_points method to get 
        # where the barrier should be placed.
        barrier_visual = VMobject(color=RED)
        barrier_visual.set_points_as_corners(barrier.get_manim_points(axes, [p.view_min, p.view_max]))
        
        # The probability curve and the wave itself. 
        prob_curve = VMobject(color=BLUE)
        wave_curve = VMobject(color=YELLOW).set_stroke(opacity=0.4)

        # Add the axes, probability curve, wave curve, and barrier to the scene.
        self.add(axes, prob_curve, wave_curve, barrier_visual)

        # --- 4. Animation Loop ---

        # Loop over 1000 steps. 
        for _ in range(1000):
            # SSFM physics step using ifft and fft.
            psi *= np.sqrt(v_oper)
            psi = ifft(sys.kinetic_oper * fft(psi))
            psi *= np.sqrt(v_oper)
            
            # Masking: We only send points to Manim that are within the view range.
            # This prevents visual artifacts at the extreme edges.
            mask = (sys.x >= p.view_min) & (sys.x <= p.view_max)
            visible_x = sys.x[mask]
            visible_prob = np.abs(psi[mask])**2
            visible_real = np.real(psi[mask])
            
            # We can then get the points for the probability and wave curves.
            p_pts = [axes.c2p(x, y) for x, y in zip(visible_x, visible_prob)]
            w_pts = [axes.c2p(x, y) for x, y in zip(visible_x, visible_real)]
            
            # We set the new values.
            prob_curve.set_points_as_corners(p_pts)
            wave_curve.set_points_as_corners(w_pts)
            
            # Stop if the wave has effectively left the building. 
            if np.sum(np.abs(psi)**2) * sys.dx < 0.0005:
                break
            
            # Wait based on frame rate.
            self.wait(1/self.camera.frame_rate)