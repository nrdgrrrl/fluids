"""GPU spatial bins with bounded storage and a substep overflow gate.

Initialize Taichi and create a ParticleState before constructing SpatialGrid.
The coordinator should call ``grid.run_substep(physics_step, ...)`` for every
substep: it rebuilds the bins, runs physics only on success, and returns the
number of omitted insertions. A nonzero result requires pausing physics until
the scene is repaired or reset. ``build()`` exposes the same count for a
coordinator that explicitly gates its own physics kernels.

Inside physics kernels, ``for_each_neighbor(i, visitor)`` visits particles
within the current smoothing radius, including i itself. The Taichi visitor
receives (i, j). Only the current cell and its 26 neighbors are searched.
Particle data stays on the GPU; only the scalar overflow count is read back.

The prescribed 256 slots are a hard limit: a finest-quality, deep initial lattice
can already exceed it (seven particles per axis fit in a 1.6 m cell). Treat that
as a recoverable overflow and choose a less dense scene before resuming.
"""

from math import isfinite
from typing import Callable

import taichi as ti

from .config import PARTICLE_SPACING, SMOOTHING_LENGTH_FACTOR
from .particles import ParticleState


GRID_SHAPE = (30, 10, 20)
CELL_SIZE = 1.6  # metres; covers 48 x 16 x 32 m from the world's lower bound
CELL_CAPACITY = 256


@ti.data_oriented
class SpatialGrid:
    """Reusable fixed allocations; cell counts include overflow attempts.

    Only slots below min(cell_counts[cell], CELL_CAPACITY) contain indices
    from the most recent build. Unused slots may contain old data. Atomic
    reservation makes insertion order unspecified, but each active index is
    inserted exactly once when no cell overflows.
    """

    def __init__(self, particles: ParticleState) -> None:
        self.particles = particles
        self._validate_scene()
        self.cell_counts = ti.field(dtype=ti.i32, shape=GRID_SHAPE)
        self.particle_indices = ti.field(dtype=ti.i32, shape=(*GRID_SHAPE, CELL_CAPACITY))
        self.overflow_count = ti.field(dtype=ti.i32, shape=())
        self._origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._radius = ti.field(dtype=ti.f32, shape=())
        self._ready = ti.field(dtype=ti.i32, shape=())

    def _validate_scene(self) -> None:
        maximum_radius = max(PARTICLE_SPACING.values()) * SMOOTHING_LENGTH_FACTOR
        radius = self.particles.smoothing_radius
        if not isfinite(radius) or not 0 < radius <= CELL_SIZE or maximum_radius > CELL_SIZE:
            raise ValueError("maximum smoothing radius must not exceed grid cell size")
        world = self.particles.world
        for lower, upper, cells in zip(world.domain_min, world.domain_max, GRID_SHAPE):
            if (
                not isfinite(lower) or not isfinite(upper)
                or not 0 < upper - lower <= cells * CELL_SIZE
            ):
                raise ValueError("world bounds must fit inside the spatial grid")

    def build(self) -> int:
        """Rebuild from the active GPU prefix and return omitted insertion count.

        Zero means every active particle was inserted. Any nonzero count makes
        the entire grid unusable for physics, even in cells that did not overflow.
        Revalidation and runtime kernel arguments support ParticleState.reset()
        without reallocating buffers or retaining compiled scene metadata.
        """
        self._validate_scene()
        self._clear(self.particles.world.domain_min, self.particles.smoothing_radius)
        self._insert()
        return int(self.overflow_count[None])

    def run_substep(self, physics_step: Callable, *args, **kwargs) -> int:
        """Build before physics; return overflow without calling physics on failure."""
        overflow = self.build()
        if overflow == 0:
            physics_step(*args, **kwargs)
        return overflow

    @ti.func
    def cell_coordinates(self, position):
        """Floor relative to the world origin, clamping boundary roundoff."""
        cell = ti.cast(ti.floor((position - self._origin[None]) / CELL_SIZE), ti.i32)
        for axis in ti.static(range(3)):
            cell[axis] = ti.max(0, ti.min(cell[axis], GRID_SHAPE[axis] - 1))
        return cell

    @ti.kernel
    def _clear(self, origin: ti.types.vector(3, ti.f32), radius: ti.f32):
        self._ready[None] = 0
        self.overflow_count[None] = 0
        self._origin[None] = origin
        self._radius[None] = radius
        for cell in ti.grouped(self.cell_counts):
            self.cell_counts[cell] = 0

    @ti.kernel
    def _insert(self):
        for i in range(self.particles.count[None]):
            cell = self.cell_coordinates(self.particles.positions[i])
            slot = ti.atomic_add(self.cell_counts[cell], 1)
            if slot < CELL_CAPACITY:
                self.particle_indices[cell[0], cell[1], cell[2], slot] = i
            else:
                ti.atomic_add(self.overflow_count[None], 1)
        self._ready[None] = ti.cast(self.overflow_count[None] == 0, ti.i32)

    @ti.func
    def for_each_neighbor(self, i, visitor: ti.template()):
        """Call a Taichi visitor for each neighbor, including self, once.

        Neighbor cells outside the grid are skipped, never clamped (which
        would visit edge cells repeatedly). Incomplete or unbuilt grids cannot
        supply partial neighbor lists; the host must gate physics via build's
        result or run_substep. The radius is uploaded on every build so a
        compiled solver follows changes in particle quality.
        """
        if self._ready[None] != 0:
            position = self.particles.positions[i]
            cell = self.cell_coordinates(position)
            for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                neighbor = cell + ti.Vector([dx, dy, dz])
                if (
                    0 <= neighbor[0] < GRID_SHAPE[0]
                    and 0 <= neighbor[1] < GRID_SHAPE[1]
                    and 0 <= neighbor[2] < GRID_SHAPE[2]
                ):
                    count = ti.min(self.cell_counts[neighbor], CELL_CAPACITY)
                    for slot in range(count):
                        j = self.particle_indices[neighbor[0], neighbor[1], neighbor[2], slot]
                        distance = self.particles.positions[j] - position
                        if distance.dot(distance) <= self._radius[None] ** 2:
                            visitor(i, j)
