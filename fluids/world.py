"""Deterministic CPU geometry shared by the renderer and collision solver.

The shoreline is x = 0: the ocean has negative x and land has positive x.
World bounds use the configured domain dimensions, translating x so the
shoreline bisects the container. As in :mod:`fluids.config`, y points up and
building heights specify their tops relative to the domain floor.

``generate_world(settings)`` returns immutable boxes and a lookup keyed by
building plot (x index, z index). ``world.triangle_mesh()`` expands those same
boxes into float32 triangle vertices and RGB colors, without initializing a GPU.
"""

from dataclasses import dataclass, field
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .config import BOUNDARY_MARGIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_SIZE, Settings


CITY_COLUMNS = 6
CITY_ROWS = 8
STREET_WIDTH = 1.0
# Stay below the smallest supported building top (0.5 m from the floor).
LAND_HEIGHT = 0.25
LAND_COLOR = (0.32, 0.40, 0.25)

Vector3 = tuple[float, float, float]
Plot = tuple[int, int]


@dataclass(frozen=True)
class Box:
    """Axis-aligned solid; land has no plot, buildings retain their grid index."""

    minimum: Vector3
    maximum: Vector3
    color: Vector3
    plot: Plot | None = None


@dataclass(frozen=True)
class World:
    """Authoritative solids and container bounds in shoreline coordinates."""

    land: Box
    buildings: tuple[Box, ...]
    domain_min: Vector3
    domain_max: Vector3
    buildings_by_plot: Mapping[Plot, Box] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "buildings_by_plot", MappingProxyType({
            building.plot: building for building in self.buildings
        }))

    @property
    def boxes(self) -> tuple[Box, ...]:
        """All collision/rendering solids, with land first."""
        return (self.land, *self.buildings)

    @property
    def plot_size(self) -> tuple[float, float]:
        """Cell dimensions for converting collision positions to plot indices."""
        return (
            self.domain_max[0] / CITY_COLUMNS,
            (self.domain_max[2] - self.domain_min[2]) / CITY_ROWS,
        )

    def triangle_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (vertices, colors), each shaped (36 * box count, 3)."""
        return boxes_to_mesh(self.boxes)


def generate_world(
    settings: Settings, *, domain_width: float = DOMAIN_SIZE[0],
    land_height: float = LAND_HEIGHT,
) -> World:
    """Generate six by eight plots with at least one metre between buildings.

    ``domain_width`` is the full ocean-plus-land extent. Supported widths leave
    positive room for a building after reserving streets in every plot. Land
    height is constant across the land half and measured from the domain floor;
    it must be below the minimum configured building top. The other container
    dimensions come from config.

    A local seeded RNG samples every plot even if vacant, so increasing density
    adds buildings without changing the geometry or colors of existing ones.
    """
    if not isfinite(domain_width) or domain_width <= 2 * CITY_COLUMNS * STREET_WIDTH:
        raise ValueError("domain_width must leave room for six plots and one-metre streets")
    if not isfinite(land_height) or not 0 < land_height < settings.building_min_height:
        raise ValueError("land_height must be positive and below building_min_height")
    if settings.building_max_height > DOMAIN_SIZE[1] - BOUNDARY_MARGIN:
        raise ValueError("building tops must fit below the container ceiling")

    half_width = domain_width / 2
    floor = DOMAIN_MIN[1]
    land_top = floor + land_height
    domain_min = (-half_width, floor, DOMAIN_MIN[2])
    domain_max = (half_width, DOMAIN_MAX[1], DOMAIN_MAX[2])
    land = Box(
        (0.0, floor, domain_min[2]),
        (half_width, land_top, domain_max[2]), LAND_COLOR,
    )
    plot_width = half_width / CITY_COLUMNS
    plot_depth = (domain_max[2] - domain_min[2]) / CITY_ROWS
    if plot_depth <= STREET_WIDTH:
        raise ValueError("domain depth must leave room for eight plots and one-metre streets")

    rng = Random(settings.city_seed)
    buildings = []
    for column in range(CITY_COLUMNS):
        for row in range(CITY_ROWS):
            occupancy = rng.random()
            # Each plot reserves half a street on every side. The size factors
            # add clearance beyond this minimum, including at narrow widths.
            width = rng.uniform(0.45, 0.95) * (plot_width - STREET_WIDTH)
            depth = rng.uniform(0.45, 0.95) * (plot_depth - STREET_WIDTH)
            top = floor + rng.uniform(
                settings.building_min_height, settings.building_max_height,
            )
            color = (rng.uniform(0.45, 0.85), rng.uniform(0.45, 0.85), rng.uniform(0.45, 0.85))
            if occupancy >= settings.city_density:
                continue
            center_x = (column + 0.5) * plot_width
            center_z = domain_min[2] + (row + 0.5) * plot_depth
            buildings.append(Box(
                (center_x - width / 2, land_top, center_z - depth / 2),
                (center_x + width / 2, top, center_z + depth / 2),
                color, (column, row),
            ))

    return World(land, tuple(buildings), domain_min, domain_max)


# Two outward-facing triangles on each of -x, +x, -y, +y, -z, +z.
# Corner numbering: bottom 0..3, top 4..7, with x varying before z.
_BOX_TRIANGLES = np.array([
    (0, 6, 4), (0, 2, 6),
    (1, 7, 3), (1, 5, 7),
    (0, 3, 2), (0, 1, 3),
    (4, 7, 5), (4, 6, 7),
    (0, 5, 1), (0, 4, 5),
    (2, 7, 6), (2, 3, 7),
], dtype=np.intp).ravel()


def boxes_to_mesh(boxes: Iterable[Box]) -> tuple[np.ndarray, np.ndarray]:
    """Expand each shared box into 12 triangles and matching per-vertex RGB.

    Both outputs are contiguous float32 arrays, suitable for GPU upload. Each
    successive group of three vertices is a triangle. Box order is preserved,
    and an empty input returns two arrays shaped (0, 3).
    """
    boxes = tuple(boxes)
    vertices = np.empty((36 * len(boxes), 3), dtype=np.float32)
    colors = np.empty_like(vertices)
    for index, box in enumerate(boxes):
        x0, y0, z0 = box.minimum
        x1, y1, z1 = box.maximum
        corners = np.array([
            (x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (x1, y0, z1),
            (x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (x1, y1, z1),
        ], dtype=np.float32)
        start = 36 * index
        vertices[start:start + 36] = corners[_BOX_TRIANGLES]
        colors[start:start + 36] = box.color
    return vertices, colors
