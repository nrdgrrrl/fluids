"""Localized GPU wave impulses in three-dimensional shoreline coordinates.

``WaveLauncher(particles).launch(settings)`` adds velocity to the active prefix
without changing water quantity, mass, positions, or other particle fields.
Each launch reads live settings; it neither resets the scene nor adds particles.
The full-width band is centered at x = -12 and spans the water's y/z extent.
A raised cosine tapers both velocity additions smoothly to zero at its edges.

Forward speed adds motion toward the city (+x). Height controls lift energy via
``sqrt(2 * max(gravity, REFERENCE_ACCELERATION) * wave_height)``; pressure,
collisions, and existing motion determine the actual crest. A small reference
acceleration keeps lift usable in zero gravity. Combined velocity, including
existing motion in all three axes, uses the integrator's MAX_SPEED safeguard.

Launch before computing forces for the next step. DynamicsSolver.launch_wave
also invalidates any previously prepared force pass. Particle state stays on
the GPU; construction requires no additional fields or particle downloads.
"""

from math import pi, sqrt

import taichi as ti

from .config import MAX_SPEED, Settings
from .particles import ParticleState


IMPULSE_CENTER_X = -12.0  # metres, offshore from the shoreline at x = 0
REFERENCE_ACCELERATION = 0.1  # m/s^2; minimum lift reference in low/zero gravity


@ti.data_oriented
class WaveLauncher:
    """Reusable additive impulses over the current active particle prefix."""

    def __init__(self, particles: ParticleState) -> None:
        self.particles = particles

    def launch(self, settings: Settings | None = None) -> None:
        """Apply current speed, width and lift without rebuilding the scene."""
        settings = self.particles.settings if settings is None else settings
        lift_speed = sqrt(
            2.0 * max(settings.gravity, REFERENCE_ACCELERATION) * settings.wave_height
        )
        if settings.wave_speed == 0.0 and lift_speed == 0.0:
            return
        self._launch(settings.wave_width, settings.wave_speed, lift_speed)

    @ti.kernel
    def _launch(self, width: ti.f32, forward_speed: ti.f32, lift_speed: ti.f32):
        for i in range(self.particles.count[None]):
            offset = self.particles.positions[i][0] - IMPULSE_CENTER_X
            if ti.abs(offset) < 0.5 * width:
                envelope = 0.5 * (1.0 + ti.cos(2.0 * pi * offset / width))
                velocity = self.particles.velocities[i] + envelope * ti.Vector(
                    [forward_speed, lift_speed, 0.0]
                )
                speed = velocity.norm()
                if speed > MAX_SPEED:
                    velocity *= MAX_SPEED / speed
                self.particles.velocities[i] = velocity
