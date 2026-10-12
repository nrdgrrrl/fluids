"""GPU poly6 density estimation and bounded, nonnegative water pressure.

Construct ``DensitySolver(grid)`` after initializing Taichi and the particles.
``compute()`` rebuilds the spatial grid before launching the density pass and
returns the overflow count. On overflow it leaves density and pressure untouched;
the coordinator must pause further physics until the scene is repaired or reset.

The same solver can be reused after ParticleState.reset(). Smoothing radius and
particle mass are passed to the kernel at runtime, never captured as scene
constants by Taichi's compiler. No particle data is downloaded by this module.
"""

from math import pi

import taichi as ti

from .config import REST_DENSITY
from .grid import SpatialGrid


# Linear weakly compressible equation of state, with an artificial sound speed
# of 20 m/s. These are interactive solver safeguards, not physical water values.
PRESSURE_STIFFNESS = 20.0**2  # pressure per unit excess density (m^2/s^2)
MAX_PRESSURE = PRESSURE_STIFFNESS * REST_DENSITY  # saturate at twice rest density


@ti.data_oriented
class DensitySolver:
    """Density/pressure pass over the grid's active particle prefix."""

    def __init__(self, grid: SpatialGrid) -> None:
        self.grid = grid
        self.particles = grid.particles

    def compute(self) -> int:
        """Rebuild, then compute only if every active particle was inserted."""
        return self.grid.run_substep(
            self._compute, self.particles.smoothing_radius, self.particles.mass,
        )

    @ti.func
    def _sum_weight(self, i, j, offset, distance_squared, total, radius_squared):
        weight = ti.max(1.0 - distance_squared / radius_squared, 0.0)
        return total + weight * weight * weight

    @ti.kernel
    def _compute(self, smoothing_radius: ti.f32, particle_mass: ti.f32):
        radius_squared = smoothing_radius * smoothing_radius
        # W(r,h) = 315/(64*pi*h^9) * (h^2-r^2)^3 for r <= h.
        # The equivalent normalized form avoids forming very small h^9 values.
        self_density = particle_mass * 315.0 / (64.0 * pi * smoothing_radius**3)
        for i in range(self.particles.count[None]):
            # Self contributes one; exact support-boundary weights are zero.
            kernel_sum = self.grid.reduce_neighbors(i, self._sum_weight, 0.0, radius_squared)
            density = self_density * kernel_sum
            self.particles.densities[i] = density
            compression = ti.max(density - REST_DENSITY, 0.0)
            self.particles.pressures[i] = ti.min(
                PRESSURE_STIFFNESS * compression, MAX_PRESSURE,
            )
