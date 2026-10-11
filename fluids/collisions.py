"""GPU contacts with the same boxes used to render the procedural world.

``CollisionSolver(particles).resolve(settings)`` corrects the active GPU prefix.
Buildings occupy fixed (column, row) slots, including an explicit empty flag;
each particle checks only its current plot, irrespective of city density.
Scene resets upload bounds again without reallocating fields or downloading
particles. Radius, container bounds and live contact settings are kernel inputs.

Boxes are expanded by the particle radius, a conservative sphere-box contact.
Penetrations project to the nearest accessible expanded face. Faces beyond the
container are inaccessible, as is a building's bottom (buried in expanded land).
Land resolves first, then buildings, so one pass also handles their overlap.
Restitution returns a fraction of incoming normal speed; friction removes a
fraction of tangential speed on impact. Departing particles keep their velocity.

The single-plot lookup requires at least 0.5 m of plot-edge clearance and a
radius no larger than 0.36 m. Validate these invariants on scene uploads so later
geometry or quality changes cannot silently invalidate collision lookup.
"""

from math import isfinite
from numbers import Integral

import numpy as np
import taichi as ti

from .config import PARTICLE_SPACING, Settings
from .particles import ParticleState
from .world import CITY_COLUMNS, CITY_ROWS, World


MAX_COLLISION_RADIUS = 0.36
MIN_PLOT_CLEARANCE = 0.5
PLOT_SHAPE = (CITY_COLUMNS, CITY_ROWS)


def _geometry_arrays(world: World) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the procedural contract before changing any GPU geometry."""
    lower, upper = np.asarray(world.domain_min), np.asarray(world.domain_max)
    land_min, land_max = np.asarray(world.land.minimum), np.asarray(world.land.maximum)
    if (
        not np.isfinite([lower, upper, land_min, land_max]).all()
        or not np.all(upper - lower > 2 * MAX_COLLISION_RADIUS)
        or not np.array_equal(land_min, [0.0, lower[1], lower[2]])
        or land_max[0] != upper[0] or land_max[2] != upper[2]
        or not lower[0] < -2 * MAX_COLLISION_RADIUS < 0 < upper[0]
        or not lower[1] < land_max[1] <= upper[1] - 2 * MAX_COLLISION_RADIUS
    ):
        raise ValueError("collision world must have procedural land and container bounds")
    width, depth = world.plot_size
    minimum = np.zeros((*PLOT_SHAPE, 3), dtype=np.float32)
    maximum = np.zeros_like(minimum)
    occupied = np.zeros(PLOT_SHAPE, dtype=np.int32)
    for building in world.buildings:
        plot = building.plot
        if (
            plot is None or len(plot) != 2
            or any(isinstance(index, bool) or not isinstance(index, Integral) for index in plot)
            or not (0 <= plot[0] < CITY_COLUMNS and 0 <= plot[1] < CITY_ROWS)
        ):
            raise ValueError("building must have a valid fixed plot slot")
        if occupied[plot]:
            raise ValueError("only one building is allowed per plot slot")
        lo, hi = np.asarray(building.minimum), np.asarray(building.maximum)
        plot_min = np.array([plot[0] * width, lower[2] + plot[1] * depth])
        plot_max = plot_min + [width, depth]
        if (
            not np.isfinite([lo, hi]).all() or not np.all(lo < hi)
            or lo[1] != land_max[1] or hi[1] > upper[1] - 2 * MAX_COLLISION_RADIUS
            or np.any(lo[[0, 2]] < plot_min + MIN_PLOT_CLEARANCE)
            or np.any(hi[[0, 2]] > plot_max - MIN_PLOT_CLEARANCE)
        ):
            raise ValueError("building must stand on land with 0.5 m plot-edge clearance and ceiling clearance")
        minimum[plot], maximum[plot], occupied[plot] = lo, hi, 1
    return minimum, maximum, occupied


@ti.data_oriented
class CollisionSolver:
    """Fixed GPU plot storage and an O(1) lookup per active particle."""

    def __init__(self, particles: ParticleState) -> None:
        self.particles = particles
        self._validate_radius()
        arrays = _geometry_arrays(particles.world)
        self.building_minimum = ti.Vector.field(3, dtype=ti.f32, shape=PLOT_SHAPE)
        self.building_maximum = ti.Vector.field(3, dtype=ti.f32, shape=PLOT_SHAPE)
        self.building_occupied = ti.field(dtype=ti.i32, shape=PLOT_SHAPE)
        self._upload(particles.world, arrays)

    def _validate_radius(self) -> None:
        radius = self.particles.collision_radius
        if (
            not isfinite(radius) or not 0 < radius <= MAX_COLLISION_RADIUS
            or max(PARTICLE_SPACING.values()) / 2 > MAX_COLLISION_RADIUS
        ):
            raise ValueError("collision radius must be positive and at most 0.36 m")

    def _upload(self, world: World, arrays) -> None:
        self.building_minimum.from_numpy(arrays[0])
        self.building_maximum.from_numpy(arrays[1])
        self.building_occupied.from_numpy(arrays[2])
        self.world = world

    def resolve(self, settings: Settings | None = None) -> None:
        """Correct positions and incoming velocities, with no particle readback."""
        self._validate_radius()
        world = self.particles.world
        if world is not self.world:
            self._upload(world, _geometry_arrays(world))
        settings = self.particles.settings if settings is None else settings
        self._resolve(
            self.particles.collision_radius, settings.collision_restitution,
            settings.collision_friction, world.domain_min, world.domain_max,
            world.land.minimum, world.land.maximum, world.plot_size,
        )

    @ti.func
    def _impact_velocity(self, velocity, normal, restitution, friction):
        normal_speed = velocity.dot(normal)
        if normal_speed < 0.0:
            tangent = velocity - normal_speed * normal
            velocity = (1.0 - friction) * tangent - restitution * normal_speed * normal
        return velocity

    @ti.func
    def _project_box(
        self, position, velocity, minimum, maximum, domain_min, domain_max,
        restitution, friction, skip_bottom: ti.template(),
    ):
        if (position >= minimum).all() and (position <= maximum).all():
            nearest = 1e30
            projected = position
            normal = ti.Vector.zero(ti.f32, 3)
            for axis in ti.static(range(3)):
                for side in ti.static(range(2)):
                    if ti.static(not (skip_bottom and axis == 1 and side == 0)):
                        face = minimum[axis]
                        direction = -1.0
                        if ti.static(side == 1):
                            face = maximum[axis]
                            direction = 1.0
                        distance = ti.abs(position[axis] - face)
                        if domain_min[axis] <= face <= domain_max[axis] and distance < nearest:
                            nearest = distance
                            projected = position
                            projected[axis] = face
                            normal = ti.Vector.zero(ti.f32, 3)
                            normal[axis] = direction
            position = projected
            velocity = self._impact_velocity(velocity, normal, restitution, friction)
        return position, velocity

    @ti.kernel
    def _resolve(
        self, radius: ti.f32, restitution: ti.f32, friction: ti.f32,
        domain_min: ti.types.vector(3, ti.f32), domain_max: ti.types.vector(3, ti.f32),
        land_min: ti.types.vector(3, ti.f32), land_max: ti.types.vector(3, ti.f32),
        plot_size: ti.types.vector(2, ti.f32),
    ):
        lower, upper = domain_min + radius, domain_max - radius
        for i in range(self.particles.count[None]):
            position, velocity = self.particles.positions[i], self.particles.velocities[i]
            # All six faces, including exact contact and arbitrarily deep escape.
            for axis in ti.static(range(3)):
                normal = ti.Vector.zero(ti.f32, 3)
                if position[axis] <= lower[axis]:
                    position[axis] = lower[axis]
                    normal[axis] = 1.0
                elif position[axis] >= upper[axis]:
                    position[axis] = upper[axis]
                    normal[axis] = -1.0
                velocity = self._impact_velocity(velocity, normal, restitution, friction)
            position, velocity = self._project_box(
                position, velocity, land_min - radius, land_max + radius,
                lower, upper, restitution, friction, False,
            )
            column = ti.cast(ti.floor(position[0] / plot_size[0]), ti.i32)
            row = ti.cast(ti.floor((position[2] - domain_min[2]) / plot_size[1]), ti.i32)
            if 0 <= column < CITY_COLUMNS and 0 <= row < CITY_ROWS:
                if self.building_occupied[column, row] != 0:
                    position, velocity = self._project_box(
                        position, velocity, self.building_minimum[column, row] - radius,
                        self.building_maximum[column, row] + radius, lower, upper,
                        restitution, friction, True,
                    )
            self.particles.positions[i], self.particles.velocities[i] = position, velocity
