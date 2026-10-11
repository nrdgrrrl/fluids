"""GPU SPH forces, semi-implicit integration, and splash colors.

Construct ``DynamicsSolver(grid)`` after initializing particles and Taichi.
``advance(frame_time, settings)`` caps elapsed time, scales it by playback speed,
and takes at most MAX_SUBSTEPS adaptive steps. Excess time and remainders below
MIN_TIME_STEP are discarded. It returns the grid overflow count; any nonzero
result requires pausing physics until the scene is repaired or reset.

For a coordinator that owns timing, ``compute_forces(settings)`` rebuilds the
grid, computes density/pressure, then writes only accelerations. On success,
``stable_time_step`` gives the current CFL, acceleration, and viscosity limit.
``integrate(dt, settings)`` consumes those forces in a separate kernel. ``step``
combines these operations for one bounded step. Integration resolves collisions
against the procedural world before the next grid rebuild.

Pressure uses -m * (p_i/rho_i^2 + p_j/rho_j^2) * grad(W_spiky), so each pair
has equal and opposite pressure acceleration before the magnitude safeguard.
Viscosity uses nu * m/rho_j * (v_j-v_i) * laplacian(W_viscosity). Density
divisors are floored at MIN_DENSITY; combined acceleration and velocity
magnitudes are capped at MAX_ACCELERATION and MAX_SPEED. The viscosity step
limit keeps explicit diffusion from reversing relative motion in a single step.
Damping multiplies the new velocity by exp(-damping * dt), then positions use
that new velocity. Fast particles below rest density blend toward pale blue.

Every force reads the same pre-integration positions and velocities. Runtime
arguments follow live settings and ParticleState.reset() without reallocations.
Only scalar overflow and time-step limits are read back; colors and all particle
state stay on the GPU. The native window remains the separate startup checkpoint.
"""

from math import isfinite, pi
from numbers import Real

import taichi as ti

from .config import (
    CFL_FACTOR, MAX_ACCELERATION, MAX_FRAME_TIME, MAX_SPEED, MAX_SUBSTEPS,
    MAX_TIME_STEP, MIN_DENSITY, MIN_TIME_STEP, NUMERICAL_EPSILON, REST_DENSITY,
    Settings,
)
from .collisions import CollisionSolver
from .density import PRESSURE_STIFFNESS, DensitySolver
from .grid import CELL_CAPACITY, GRID_SHAPE, SpatialGrid
from .particles import WATER_COLOR
from .waves import WaveLauncher


SPLASH_COLOR = (0.65, 0.88, 1.0)
SPLASH_SPEED = 5.0  # m/s at which the speed contribution saturates


def _validate_time(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@ti.data_oriented
class DynamicsSolver:
    """Reusable density, force, and integration passes over the active prefix."""

    def __init__(self, grid: SpatialGrid) -> None:
        self.grid = grid
        self.particles = grid.particles
        self.density = DensitySolver(grid)
        self.collisions = CollisionSolver(self.particles)
        self.waves = WaveLauncher(self.particles)
        self._step_limit = ti.field(dtype=ti.f32, shape=())
        self._forces_ready = False

    def launch_wave(self, settings: Settings | None = None) -> None:
        """Add a live offshore impulse; recompute forces before integrating it."""
        self.waves.launch(settings)
        self._forces_ready = False

    @property
    def stable_time_step(self) -> float:
        """Recommended maximum dt for the most recent successful force pass."""
        if not self._forces_ready:
            raise RuntimeError("compute_forces must succeed before using the time-step limit")
        return max(MIN_TIME_STEP, min(MAX_TIME_STEP, float(self._step_limit[None])))

    def compute_forces(self, settings: Settings | None = None) -> int:
        """Compute from one snapshot; leave all dynamics untouched on overflow."""
        settings = self.particles.settings if settings is None else settings
        self._forces_ready = False
        overflow = self.density.compute()
        if overflow == 0:
            self._compute_forces(
                self.particles.smoothing_radius, self.particles.mass,
                settings.viscosity, settings.gravity,
            )
            self._forces_ready = True
        return overflow

    def integrate(self, dt: float, settings: Settings | None = None) -> None:
        """Consume one force pass, rejecting oversized or unprepared steps."""
        dt = _validate_time(dt, "time step")
        if not MIN_TIME_STEP <= dt <= MAX_TIME_STEP:
            raise ValueError("time step must be within the configured bounds")
        if dt > self.stable_time_step * (1.0 + 1e-6):
            raise ValueError("time step exceeds the current stability limit")
        settings = self.particles.settings if settings is None else settings
        self._integrate(dt, settings.damping)
        self._forces_ready = False
        self.collisions.resolve(settings)

    def step(self, dt: float, settings: Settings | None = None) -> int:
        """Rebuild, compute forces, integrate and collide unless the grid overflows."""
        dt = _validate_time(dt, "time step")
        if not MIN_TIME_STEP <= dt <= MAX_TIME_STEP:
            raise ValueError("time step must be within the configured bounds")
        overflow = self.compute_forces(settings)
        if overflow == 0:
            self.integrate(dt, settings)
        return overflow

    def advance(self, frame_time: float, settings: Settings | None = None) -> int:
        """Advance a bounded frame, colliding each step; zero speed leaves fields alone."""
        frame_time = _validate_time(frame_time, "frame time")
        if frame_time < 0:
            raise ValueError("frame time must be nonnegative")
        settings = self.particles.settings if settings is None else settings
        remaining = min(frame_time, MAX_FRAME_TIME) * settings.playback_speed
        for _ in range(MAX_SUBSTEPS):
            if remaining < MIN_TIME_STEP:
                break
            overflow = self.compute_forces(settings)
            if overflow:
                return overflow
            dt = min(remaining, self.stable_time_step)
            self.integrate(dt, settings)
            remaining -= dt
        return 0

    @ti.kernel
    def _compute_forces(
        self, smoothing_radius: ti.f32, particle_mass: ti.f32,
        viscosity: ti.f32, gravity: ti.f32,
    ):
        self._step_limit[None] = MAX_TIME_STEP
        # Normalized viscosity Laplacian is 45/pi/h^5 * (1-r/h).
        # Spiky gradient magnitude is 45/pi/h^4 * (1-r/h)^2.
        kernel_scale = 45.0 / (pi * smoothing_radius**5)
        for i in range(self.particles.count[None]):
            position = self.particles.positions[i]
            velocity = self.particles.velocities[i]
            density_i = ti.max(self.particles.densities[i], MIN_DENSITY)
            pressure_i = self.particles.pressures[i] / (density_i * density_i)
            acceleration = ti.Vector([0.0, -gravity, 0.0])
            diffusion_rate = 0.0
            cell = self.grid.cell_coordinates(position)
            for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                neighbor = cell + ti.Vector([dx, dy, dz])
                if (
                    0 <= neighbor[0] < GRID_SHAPE[0]
                    and 0 <= neighbor[1] < GRID_SHAPE[1]
                    and 0 <= neighbor[2] < GRID_SHAPE[2]
                ):
                    count = ti.min(self.grid.cell_counts[neighbor], CELL_CAPACITY)
                    for slot in range(count):
                        j = self.grid.particle_indices[neighbor[0], neighbor[1], neighbor[2], slot]
                        offset = position - self.particles.positions[j]
                        distance_squared = offset.dot(offset)
                        if j != i and distance_squared < smoothing_radius**2:
                            distance = ti.sqrt(distance_squared)
                            weight = 1.0 - distance / smoothing_radius
                            density_j = ti.max(self.particles.densities[j], MIN_DENSITY)
                            if distance > NUMERICAL_EPSILON:
                                pressure_j = self.particles.pressures[j] / (density_j * density_j)
                                acceleration += (
                                    particle_mass * (pressure_i + pressure_j)
                                    * kernel_scale * smoothing_radius * weight**2 * offset / distance
                                )
                            # Coincident particles have no defined pressure
                            # direction, but still exchange velocity via viscosity.
                            rate = viscosity * particle_mass / density_j * kernel_scale * weight
                            diffusion_rate += rate
                            acceleration += rate * (self.particles.velocities[j] - velocity)
            magnitude = acceleration.norm()
            if magnitude > MAX_ACCELERATION:
                acceleration *= MAX_ACCELERATION / magnitude
            self.particles.accelerations[i] = acceleration
            # Artificial sound speed comes from the density equation of state.
            speed = ti.min(velocity.norm(), MAX_SPEED)
            dt = CFL_FACTOR * smoothing_radius / (PRESSURE_STIFFNESS**0.5 + speed)
            bounded_magnitude = ti.min(magnitude, MAX_ACCELERATION)
            dt = ti.min(dt, ti.sqrt(CFL_FACTOR * smoothing_radius / ti.max(bounded_magnitude, NUMERICAL_EPSILON)))
            if diffusion_rate > NUMERICAL_EPSILON:
                dt = ti.min(dt, 0.5 / diffusion_rate)
            ti.atomic_min(self._step_limit[None], dt)

    @ti.kernel
    def _integrate(self, dt: ti.f32, damping: ti.f32):
        for i in range(self.particles.count[None]):
            velocity = (
                self.particles.velocities[i] + self.particles.accelerations[i] * dt
            ) * ti.exp(-damping * dt)
            speed = velocity.norm()
            if speed > MAX_SPEED:
                velocity *= MAX_SPEED / speed
            self.particles.velocities[i] = velocity
            self.particles.positions[i] += velocity * dt
            speed_fraction = ti.min(speed / SPLASH_SPEED, 1.0)
            density_fraction = ti.max(0.0, 1.0 - self.particles.densities[i] / REST_DENSITY)
            blend = ti.min(speed_fraction * density_fraction, 1.0)
            self.particles.colors[i] = (
                (1.0 - blend) * ti.Vector(WATER_COLOR) + blend * ti.Vector(SPLASH_COLOR)
            )
