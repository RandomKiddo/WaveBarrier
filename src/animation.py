import numpy as np
from manim import *
from dataclasses import dataclass 
from system import QuantumSystem
from scipy.fft import fft, ifft
from potentials import *


@dataclass(frozen=True)
class SystemParams:
    nx: int = 2048
    # Visible range for the camera
    view_min: int = -10
    view_max: int = 10
    # Larger numerical range to hide the sponge "off-screen"
    sim_min: int = -13
    sim_max: int = 13
    dt: float = 0.01
    mass: float = 1.0 
    V_type: Potential = SemiCircularBarrier(radius=1.0)

# --- ANIMATION SCENE ---

class QuantumAnimation(Scene):
    params: SystemParams = SystemParams()

    def construct(self):
        p = self.params
        # Solver uses the larger SIM range
        sys = QuantumSystem((p.sim_min, p.sim_max), p.nx, p.dt, mass=p.mass)

        barrier = p.V_type
        
        # 1. Physics Setup (Real Potential + Off-screen Sponge)
        v_real = barrier.get_V(sys.x)
        
        # Place sponges strictly outside the VIEW range [-10, 10]
        # They start at +/- 10.5 and ramp up toward the SIM edges +/- 13
        strength = 150
        v_imag_l = np.where(sys.x < -10.5, strength * (sys.x + 10.5)**2, 0)
        v_imag_r = np.where(sys.x > 10.5, strength * (sys.x - 10.5)**2, 0)
        
        v_total = v_real - 1j * (v_imag_l + v_imag_r)
        v_oper = np.exp(-1j * v_total * sys.dt)

        # 2. Initial Wavepacket
        x0, sigma, k0 = -6, 0.6, 10
        psi = np.exp(-0.5 * ((sys.x - x0) / sigma)**2) * np.exp(1j * k0 * sys.x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * sys.dx)

        # 3. Visualization (Axes use the VIEW range)
        axes = Axes(x_range=[p.view_min, p.view_max, 1], y_range=[-2, 5, 1], 
                    axis_config={"include_tip": False})
        
        barrier_visual = VMobject(color=RED)
        barrier_visual.set_points_as_corners(barrier.get_manim_points(axes, [p.view_min, p.view_max]))
        
        prob_curve = VMobject(color=BLUE)
        wave_curve = VMobject(color=YELLOW).set_stroke(opacity=0.4)

        self.add(axes, prob_curve, wave_curve, barrier_visual)

        # 4. Animation Loop
        for _ in range(1000):
            # SSFM Physics Step
            psi *= np.sqrt(v_oper)
            psi = ifft(sys.kinetic_oper * fft(psi))
            psi *= np.sqrt(v_oper)
            
            # Masking: We only send points to Manim that are within the VIEW range
            # This prevents visual artifacts at the extreme edges
            mask = (sys.x >= p.view_min) & (sys.x <= p.view_max)
            visible_x = sys.x[mask]
            visible_prob = np.abs(psi[mask])**2
            visible_real = np.real(psi[mask])
            
            p_pts = [axes.c2p(x, y) for x, y in zip(visible_x, visible_prob)]
            w_pts = [axes.c2p(x, y) for x, y in zip(visible_x, visible_real)]
            
            prob_curve.set_points_as_corners(p_pts)
            wave_curve.set_points_as_corners(w_pts)
            
            # Stop if the wave has effectively left the building
            if np.sum(np.abs(psi)**2) * sys.dx < 0.0005:
                break
            
            self.wait(1/self.camera.frame_rate)