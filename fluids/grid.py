"""GPU spatial bins with bounded storage and a substep overflow gate.

Initialize Taichi and create a ParticleState before constructing SpatialGrid.
The coordinator should call ``grid.run_substep(physics_step, ...)`` for every
substep: it rebuilds the bins, runs physics only on success, and returns the
number of omitted insertions. A nonzero result requires pausing physics until
the scene is repaired or reset. ``build()`` exposes the same count for a
coordinator that explicitly gates its own physics kernels.

Inside physics kernels, ``for_each_neighbor(i, visitor)`` visits particles
within the current smoothing radius, including i itself. The Taichi visitor
receives (i, j). The default layout searches the current cell and its 26 neighbors.
Particle data stays on the GPU; only the scalar overflow count is read back.

The 256 slots remain a hard limit. Compression can overflow a cell; treat that
as a recoverable failure and repair or reset the scene before resuming.
"""

from math import ceil, isfinite, prod
from typing import Callable

import taichi as ti

from .config import PARTICLE_SPACING, SMOOTHING_LENGTH_FACTOR
from .particles import ParticleState


# Preserve the old layout's supported bounds, including wider test worlds.
GRID_EXTENT = (48.0, 16.0, 32.0)
CELL_SIZE = 1.0  # metres; the maximum supported smoothing radius
GRID_SHAPE = tuple(ceil(extent / CELL_SIZE) for extent in GRID_EXTENT)
CELL_CAPACITY = 256


@ti.data_oriented
class SpatialGrid:
    """Reusable fixed allocations; cell counts include overflow attempts.

    Only slots below min(cell_counts[cell], CELL_CAPACITY) contain indices
    from the most recent build. Unused slots may contain old data. Atomic
    reservation makes insertion order unspecified, but each active index is
    inserted exactly once when no cell overflows.
    """

    def __init__(self, particles: ParticleState, *, cell_size: float = CELL_SIZE) -> None:
        self.particles = particles
        minimum_radius = min(PARTICLE_SPACING.values()) * SMOOTHING_LENGTH_FACTOR
        if not isfinite(cell_size) or cell_size < minimum_radius:
            raise ValueError("grid cell size must be at least the minimum smoothing radius")
        # Layout is immutable for this allocation. Alternate widths allow a fair
        # benchmark; smaller cells also support later quality resets by widening
        # the stencil, without allocating fields again.
        self.cell_size = float(cell_size)
        self.shape = tuple(ceil(extent / self.cell_size) for extent in GRID_EXTENT)
        self.max_span = ceil(max(PARTICLE_SPACING.values()) * SMOOTHING_LENGTH_FACTOR / self.cell_size)
        self._validate_scene()
        self.cell_counts = ti.field(dtype=ti.i32, shape=self.shape)
        self.particle_indices = ti.field(dtype=ti.i32, shape=(*self.shape, CELL_CAPACITY))
        self.overflow_count = ti.field(dtype=ti.i32, shape=())
        self._origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._radius = ti.field(dtype=ti.f32, shape=())
        self._span = ti.field(dtype=ti.i32, shape=())
        self._ready = ti.field(dtype=ti.i32, shape=())

    @property
    def storage_bytes(self) -> int:
        """Logical field bytes, excluding backend padding and driver allocations."""
        return 4 * (prod(self.shape) * (CELL_CAPACITY + 1) + 7)

    def _validate_scene(self) -> None:
        maximum_radius = max(PARTICLE_SPACING.values()) * SMOOTHING_LENGTH_FACTOR
        radius = self.particles.smoothing_radius
        if (
            not isfinite(radius) or not 0 < radius <= maximum_radius
            or maximum_radius > self.max_span * self.cell_size
        ):
            raise ValueError("smoothing radius must fit inside the grid search stencil")
        world = self.particles.world
        for lower, upper, extent in zip(world.domain_min, world.domain_max, GRID_EXTENT):
            if (
                not isfinite(lower) or not isfinite(upper)
                or not 0 < upper - lower <= extent
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

    def reset(self) -> None:
        """Clear bins and invalidate neighbors after a scene reset, in place."""
        self._validate_scene()
        self._clear(self.particles.world.domain_min, self.particles.smoothing_radius)

    def run_substep(self, physics_step: Callable, *args, **kwargs) -> int:
        """Build before physics; return overflow without calling physics on failure."""
        overflow = self.build()
        if overflow == 0:
            physics_step(*args, **kwargs)
        return overflow

    @ti.func
    def cell_coordinates(self, position):
        """Floor relative to the world origin, clamping boundary roundoff."""
        cell = ti.cast(ti.floor((position - self._origin[None]) / self.cell_size), ti.i32)
        for axis in ti.static(range(3)):
            cell[axis] = ti.max(0, ti.min(cell[axis], self.shape[axis] - 1))
        return cell

    @ti.kernel
    def _clear(self, origin: ti.types.vector(3, ti.f32), radius: ti.f32):
        self._ready[None] = 0
        self.overflow_count[None] = 0
        self._origin[None] = origin
        self._radius[None] = radius
        self._span[None] = ti.cast(ti.ceil(radius / self.cell_size), ti.i32)
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
    def reduce_neighbors(self, i, visitor: ti.template(), initial, context: ti.template()):
        """Reduce complete neighbors using an inlined Taichi visitor.

        Neighbor cells outside the grid are skipped, never clamped (which
        would visit edge cells repeatedly). Incomplete or unbuilt grids cannot
        supply partial neighbor lists; the host must gate physics via build's
        result or run_substep. The radius is uploaded on every build so a
        compiled solver follows changes in particle quality.
        The visitor receives (i, j, j-minus-i offset, squared distance, accumulator,
        context) and returns the next accumulator. Density and force kernels use
        this same traversal so shape, cell width, and stencil have one authority.
        """
        result = initial
        if self._ready[None] != 0:
            position = self.particles.positions[i]
            cell = self.cell_coordinates(position)
            span = 1
            if ti.static(self.max_span > 1):
                span = self._span[None]
            for dx, dy, dz in ti.ndrange(
                (-span, span + 1), (-span, span + 1), (-span, span + 1),
            ):
                neighbor = cell + ti.Vector([dx, dy, dz])
                if (
                    0 <= neighbor[0] < self.shape[0]
                    and 0 <= neighbor[1] < self.shape[1]
                    and 0 <= neighbor[2] < self.shape[2]
                ):
                    count = ti.min(self.cell_counts[neighbor], CELL_CAPACITY)
                    for slot in range(count):
                        j = self.particle_indices[neighbor[0], neighbor[1], neighbor[2], slot]
                        distance = self.particles.positions[j] - position
                        squared = distance.dot(distance)
                        if squared <= self._radius[None] ** 2:
                            result = visitor(i, j, distance, squared, result, context)
        return result

    @ti.func
    def _visit(self, i, j, offset, squared, result, visitor: ti.template()):
        visitor(i, j)
        return result

    @ti.func
    def for_each_neighbor(self, i, visitor: ti.template()):
        """Visit every neighbor, including self, once; empty on failed builds."""
        self.reduce_neighbors(i, self._visit, 0, visitor)
