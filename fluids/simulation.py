"""Lifecycle and bounded playback for the GPU simulation.

Initialize Taichi, then construct one ``Simulation(settings)``. ``reset`` is
the only operation that applies structural settings; ``requires_reset`` lets
the UI compare requested settings with the applied scene. Pass live settings
to ``advance(frame_time, settings)`` each rendered frame. Frame time is elapsed
wall time since the previous frame, including while paused, so resuming never
includes the entire paused interval. ``step(settings)`` advances exactly one
adaptive physics timestep, even while paused or at zero playback speed.

Only complete physics timesteps consume the accumulator. Fractional time carries
between frames; frame spikes and work beyond MAX_SUBSTEPS are discarded and
reported separately from simulated time. Catch SpatialGridOverflowError in the
UI and pause until the scene is repaired or reset. Failed substeps consume no
simulation time and run no density, force, integration, or collision kernels.
Particle, bin, and collision fields are allocated once and reused on every reset.
"""

from .config import DEFAULT_SETTINGS, MAX_FRAME_TIME, MAX_SUBSTEPS, MIN_TIME_STEP, Settings
from .dynamics import DynamicsSolver, _validate_time
from .grid import CELL_SIZE, SpatialGrid
from .particles import PARTICLE_CAPACITY, ParticleState, plan_water_lattice
from .world import generate_world


class SpatialGridOverflowError(RuntimeError):
    """Recoverable incomplete neighbor grid; repair or reset before resuming."""

    def __init__(self, overflow_count: int) -> None:
        self.overflow_count = overflow_count
        super().__init__(
            f"Spatial bins overflowed: {overflow_count} particle insertions were "
            "omitted. Physics stopped; reduce particle quality or water depth "
            "and reset the simulation."
        )


class Simulation:
    """Single owner of scene resources, physics pass order, and playback time.

    ``advance`` returns the number of completed substeps. ``simulated_time`` and
    ``discarded_time`` are seconds of simulation time (after playback scaling).
    Paused wall time is never requested simulation time and is not discarded
    catch-up time. ``accumulated_time`` exposes the small unconsumed remainder.
    Reset clears timing and errors but preserves the user's pause selection.
    """

    def __init__(
        self, settings: Settings = DEFAULT_SETTINGS, *, capacity: int = PARTICLE_CAPACITY,
        grid_cell_size: float = CELL_SIZE,
    ) -> None:
        self.world = generate_world(settings)
        self.particles = ParticleState(settings, self.world, capacity=capacity)
        self.grid = SpatialGrid(self.particles, cell_size=grid_cell_size)
        self.dynamics = DynamicsSolver(self.grid)
        self.applied_settings = settings
        self._paused = False
        self._clear_time()
        self.dynamics.reset()

    def _clear_time(self) -> None:
        self.simulated_time = 0.0
        self.discarded_time = 0.0
        self.accumulated_time = 0.0
        self.last_time_step = 0.0
        self.last_substeps = 0
        self.last_error: SpatialGridOverflowError | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = bool(value)
        if self._paused:
            self._discard_backlog()

    def requires_reset(self, settings: Settings) -> bool:
        """Whether the requested structural controls differ from this scene."""
        return settings.reset_signature() != self.applied_settings.reset_signature()

    def reset(self, settings: Settings | None = None) -> None:
        """Regenerate world and water together and reset timing, without allocation.

        A capacity failure is checked before changing the existing scene. With
        no arguments, reproduce the last applied scene, including its seed.
        Collision geometry is uploaded now, including when playback is paused.
        """
        settings = self.applied_settings if settings is None else settings
        world = generate_world(settings)
        plan_water_lattice(settings, world, capacity=self.particles.capacity)
        self.particles.reset(settings, world)
        self.dynamics.reset()
        self.world = world
        self.applied_settings = settings
        self._clear_time()

    def launch_wave(self, settings: Settings | None = None) -> None:
        """Apply one live impulse to existing water, including while paused."""
        self.dynamics.launch_wave(self.applied_settings if settings is None else settings)

    def _discard_backlog(self) -> None:
        self.discarded_time += self.accumulated_time
        self.accumulated_time = 0.0

    def _prepare_step(self, settings: Settings) -> float:
        # compute_forces owns grid -> density -> forces, and gates all physics
        # on the grid's scalar overflow result. No particle readback is needed.
        overflow = self.dynamics.compute_forces(settings)
        if overflow:
            self._discard_backlog()
            self.last_error = SpatialGridOverflowError(overflow)
            raise self.last_error
        self.last_error = None
        return self.dynamics.stable_time_step

    def _integrate(self, dt: float, settings: Settings) -> None:
        # Integration consumes the prepared forces and resolves collisions.
        self.dynamics.integrate(dt, settings)
        self.simulated_time += dt
        self.last_time_step = dt
        self.last_substeps += 1

    def step(self, settings: Settings | None = None) -> float:
        """Take exactly one physics timestep; ignore pause and playback speed."""
        settings = self.applied_settings if settings is None else settings
        self.last_substeps = 0
        self._discard_backlog()
        dt = self._prepare_step(settings)
        self._integrate(dt, settings)
        return dt

    def advance(self, frame_time: float, settings: Settings | None = None) -> int:
        """Accumulate elapsed time and run at most MAX_SUBSTEPS complete steps.

        Timestep selection depends on physics, never the remaining frame time.
        At the work limit all remaining requested time is dropped, so overloaded
        frames visibly slow playback without accumulating a catch-up backlog.
        """
        frame_time = _validate_time(frame_time, "frame time")
        if frame_time < 0:
            raise ValueError("frame time must be nonnegative")
        settings = self.applied_settings if settings is None else settings
        self.last_substeps = 0
        if self.paused or settings.playback_speed == 0:
            self._discard_backlog()
            return 0
        accepted_time = min(frame_time, MAX_FRAME_TIME)
        self.discarded_time += (frame_time - accepted_time) * settings.playback_speed
        self.accumulated_time += accepted_time * settings.playback_speed
        for _ in range(MAX_SUBSTEPS):
            if self.accumulated_time < MIN_TIME_STEP:
                break
            dt = self._prepare_step(settings)
            if self.accumulated_time + 1e-12 < dt:
                break
            self._integrate(dt, settings)
            self.accumulated_time = max(0.0, self.accumulated_time - dt)
        if self.last_substeps == MAX_SUBSTEPS:
            self._discard_backlog()
        return self.last_substeps
