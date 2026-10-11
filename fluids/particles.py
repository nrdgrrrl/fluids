"""Bounded particle fields and reset-only CPU seeding in shoreline coordinates.

Initialize Taichi before constructing ``ParticleState(settings, world)``. Pass
``positions``, ``colors``, ``active_count`` and ``collision_radius`` directly to
Renderer.render's particle arguments. Solvers can iterate over ``count[None]``
on the GPU; the host active count is reset metadata, requiring no readback.

Reset reuses every field and discards its temporary float32 initialization
array. There is no particle download in this module.
"""

from dataclasses import dataclass
from math import floor, isfinite, prod
from numbers import Integral

import numpy as np
import taichi as ti

from .config import BOUNDARY_MARGIN, REST_DENSITY, SMOOTHING_LENGTH_FACTOR, Settings
from .world import World, generate_world


# The deepest, finest supported ocean needs 114,000 particles. Reserve some
# headroom, but never grow these allocations during a reset.
PARTICLE_CAPACITY = 131_072
COASTLINE_GAP = 1.0  # metres, independent of depth and quality
WATER_COLOR = (0.04, 0.48, 0.85)


def _validate_capacity(capacity: int) -> int:
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, Integral)
        or not 0 < capacity < 2**31
    ):
        raise ValueError("particle capacity must be a positive 32-bit integer")
    return int(capacity)


@dataclass(frozen=True)
class WaterLattice:
    """Small CPU metadata describing a regular, centered 3D lattice.

    Particles represent equal shares of the requested water volume. The mass
    starts from rest_density * spacing**3 and accounts for the partial cells
    left at the edges. This preserves water quantity across quality changes,
    including shallow water whose depth is only one or two particle spacings.
    """

    spacing: float
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    volume: float

    @property
    def count(self) -> int:
        return prod(self.shape)

    @property
    def smoothing_radius(self) -> float:
        return SMOOTHING_LENGTH_FACTOR * self.spacing

    @property
    def collision_radius(self) -> float:
        return 0.5 * self.spacing

    @property
    def mass(self) -> float:
        cell_volume = self.spacing**3
        edge_correction = self.volume / (self.count * cell_volume)
        return REST_DENSITY * cell_volume * edge_correction

    def positions(self) -> np.ndarray:
        """Create only the active positions, not a capacity-sized staging buffer."""
        positions = np.empty((self.count, 3), dtype=np.float32)
        grid = positions.reshape(*self.shape, 3)
        for axis, size in enumerate(self.shape):
            broadcast_shape = [1, 1, 1]
            broadcast_shape[axis] = size
            coordinates = self.origin[axis] + self.spacing * np.arange(size, dtype=np.float64)
            grid[..., axis] = coordinates.reshape(broadcast_shape)
        return positions


def plan_water_lattice(
    settings: Settings, world: World, *, capacity: int = PARTICLE_CAPACITY,
) -> WaterLattice:
    """Validate dimensions and capacity before creating arrays or touching GPU state.

    Water occupies the ocean half up to one metre before the land, with wall
    clearance in x and z. Depth is measured from the domain floor. Centering
    complete cells in this fixed volume keeps collision spheres inside it.
    City settings do not affect the initial water quantity.
    """
    capacity = _validate_capacity(capacity)
    spacing = settings.particle_spacing
    lower = (
        world.domain_min[0] + BOUNDARY_MARGIN,
        world.domain_min[1],
        world.domain_min[2] + BOUNDARY_MARGIN,
    )
    upper = (
        world.land.minimum[0] - COASTLINE_GAP,
        world.domain_min[1] + settings.water_depth,
        world.domain_max[2] - BOUNDARY_MARGIN,
    )
    if upper[1] > world.domain_max[1] - BOUNDARY_MARGIN:
        raise ValueError("initial water depth must fit below the container ceiling")
    lengths = tuple(hi - lo for lo, hi in zip(lower, upper))
    if any(not isfinite(length) or length < spacing for length in lengths):
        raise ValueError("ocean volume must fit at least one particle cell on each axis")
    # Tolerance only avoids losing a layer to roundoff at exact multiples.
    shape = tuple(floor(length / spacing + 1e-9) for length in lengths)
    count = prod(shape)
    if count > capacity:
        raise ValueError(f"Initial water needs {count} particles; capacity is {capacity}")
    origin = tuple(
        lo + (length - (size - 1) * spacing) / 2
        for lo, length, size in zip(lower, lengths, shape)
    )
    return WaterLattice(spacing, shape, origin, prod(lengths))


@ti.data_oriented
class ParticleState:
    """Six fixed-capacity GPU fields, with a separately tracked active prefix.

    ``reset()`` restores the saved scene. Pass new settings and a new world
    together to change the scene; omitted arguments retain their saved values.
    Oversized proposals leave both field contents and scene metadata unchanged.
    Density and pressure start at zero for the solver to compute from positions.
    """

    def __init__(
        self, settings: Settings, world: World | None = None, *,
        capacity: int = PARTICLE_CAPACITY,
    ) -> None:
        self.capacity = _validate_capacity(capacity)
        world = generate_world(settings) if world is None else world
        lattice = plan_water_lattice(settings, world, capacity=self.capacity)
        # Even the initial capacity check and CPU staging precede GPU allocation.
        positions = lattice.positions()
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.capacity)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.capacity)
        self.accelerations = ti.Vector.field(3, dtype=ti.f32, shape=self.capacity)
        self.densities = ti.field(dtype=ti.f32, shape=self.capacity)
        self.pressures = ti.field(dtype=ti.f32, shape=self.capacity)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.capacity)
        self.count = ti.field(dtype=ti.i32, shape=())
        self._reset_fields(positions, lattice.count)
        self.settings, self.world, self.lattice = settings, world, lattice
        self.active_count = lattice.count

    @property
    def spacing(self) -> float:
        return self.lattice.spacing

    @property
    def smoothing_radius(self) -> float:
        return self.lattice.smoothing_radius

    @property
    def collision_radius(self) -> float:
        return self.lattice.collision_radius

    @property
    def mass(self) -> float:
        return self.lattice.mass

    def reset(self, settings: Settings | None = None, world: World | None = None) -> None:
        """Validate and upload a fresh lattice; clear all fields, including padding."""
        settings = self.settings if settings is None else settings
        world = self.world if world is None else world
        lattice = plan_water_lattice(settings, world, capacity=self.capacity)
        positions = lattice.positions()
        self._reset_fields(positions, lattice.count)
        self.settings, self.world, self.lattice = settings, world, lattice
        self.active_count = lattice.count

    @ti.kernel
    def _reset_fields(self, initial: ti.types.ndarray(dtype=ti.f32, ndim=2), count: ti.i32):
        for i in range(self.capacity):
            self.positions[i] = ti.Vector.zero(ti.f32, 3)
            self.velocities[i] = ti.Vector.zero(ti.f32, 3)
            self.accelerations[i] = ti.Vector.zero(ti.f32, 3)
            self.densities[i] = 0.0
            self.pressures[i] = 0.0
            self.colors[i] = ti.Vector.zero(ti.f32, 3)
            if i < count:
                self.positions[i] = ti.Vector([initial[i, 0], initial[i, 1], initial[i, 2]])
                self.colors[i] = ti.Vector(WATER_COLOR)
        self.count[None] = count
