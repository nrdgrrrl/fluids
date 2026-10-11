"""Reusable Taichi scene geometry, in the world's y-up shoreline coordinates.

Initialize Taichi before constructing a renderer. The caller owns the window,
camera, and presentation; this module submits geometry to a GGUI scene:

    renderer = Renderer(generate_world(settings))
    scene = window.get_scene()
    # Each frame, after updating an orbit camera:
    scene.set_camera(camera)
    renderer.render(scene)
    window.get_canvas().scene(scene)
    window.show()

Call ``renderer.update_world(new_world)`` after a city reset. Once a solver
exists, pass its GPU position field as ``particle_positions`` and its active
count as ``particle_count``. No particle positions are downloaded to the CPU.
This scene interface is separate from bootstrap's wgpu clear-window canvas.
"""

from math import isfinite
from numbers import Integral

import numpy as np
import taichi as ti

from .world import World


# A full 6 x 8 city, land, and floor need only 590 triangles. Keep spare room
# without allocating new fields when density or the city seed changes.
STATIC_VERTEX_CAPACITY = 3 * 2048
OUTLINE_VERTEX_COUNT = 24  # two endpoints for each of the twelve box edges
FLOOR_COLOR = (0.12, 0.24, 0.31)
OUTLINE_COLOR = (0.72, 0.82, 0.90)
WATER_COLOR = (0.04, 0.48, 0.85)
AMBIENT_LIGHT = (0.45, 0.45, 0.50)


def _container_outline(world: World) -> np.ndarray:
    """Return the bottom ring, top ring, and four upright edges."""
    x0, y0, z0 = world.domain_min
    x1, y1, z1 = world.domain_max
    corners = np.array([
        (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
        (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1),
    ], dtype=np.float32)
    edges = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ], dtype=np.intp)
    return np.ascontiguousarray(corners[edges.ravel()])


def _static_mesh(world: World) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad the floor and solid geometry with invisible degenerate triangles."""
    vertex_count = 36 * len(world.boxes) + 6
    if vertex_count > STATIC_VERTEX_CAPACITY:
        raise ValueError(
            f"World needs {vertex_count} vertices; renderer capacity is "
            f"{STATIC_VERTEX_CAPACITY}"
        )

    vertices = np.zeros((STATIC_VERTEX_CAPACITY, 3), dtype=np.float32)
    colors = np.zeros_like(vertices)
    # Supply finite normals even for padding, avoiding automatic normalization
    # of zero-area triangles in GGUI and per-frame normal generation.
    normals = np.zeros_like(vertices)
    normals[:, 1] = 1.0
    solid_vertices, solid_colors = world.triangle_mesh()
    vertices[:vertex_count - 6] = solid_vertices
    colors[:vertex_count - 6] = solid_colors

    x0, y0, z0 = world.domain_min
    x1, _, z1 = world.domain_max
    # Upward-facing floor only: no opaque container walls or ceiling.
    vertices[vertex_count - 6:vertex_count] = [
        (x0, y0, z0), (x0, y0, z1), (x1, y0, z1),
        (x0, y0, z0), (x1, y0, z1), (x1, y0, z0),
    ]
    colors[vertex_count - 6:vertex_count] = FLOOR_COLOR

    triangles = vertices[:vertex_count].reshape(-1, 3, 3)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    np.divide(face_normals, lengths, out=face_normals, where=lengths > 0)
    normals[:vertex_count] = np.repeat(face_normals, 3, axis=0)
    return vertices, colors, normals, vertex_count


class Renderer:
    """Fixed GPU fields shared by every world reset and frame.

    Construct once after ``initialize_gpu()``. ``update_world`` performs CPU
    mesh generation and uploads only for changed geometry. ``render`` reads
    existing GPU fields; the scene's own staging stays on the GPU with Vulkan.
    The camera belongs to the caller so orbit controls remain independent.
    """

    def __init__(self, world: World) -> None:
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.outline_vertices = ti.Vector.field(3, dtype=ti.f32, shape=OUTLINE_VERTEX_COUNT)
        self.world: World | None = None
        self.vertex_count = 0
        self.update_world(world)

    def update_world(self, world: World) -> bool:
        """Upload changed static geometry in place; return whether it changed.

        Equal regenerated worlds also skip uploads. A smaller city overwrites
        the entire mesh, making all leftover triangles degenerate. Container
        edges are uploaded only when the bounds change.
        """
        if world == self.world:
            return False
        vertices, colors, normals, vertex_count = _static_mesh(world)
        self.vertices.from_numpy(vertices)
        self.colors.from_numpy(colors)
        self.normals.from_numpy(normals)
        if self.world is None or (
            world.domain_min != self.world.domain_min
            or world.domain_max != self.world.domain_max
        ):
            self.outline_vertices.from_numpy(_container_outline(world))
        self.world = world
        self.vertex_count = vertex_count
        return True

    def render(
        self, scene, *, particle_positions=None, particle_radius: float = 0.12,
        particle_count: int | None = None, particle_colors=None,
    ) -> None:
        """Submit lit, two-sided solids, the box outline, and optional water.

        ``scene`` is obtained with ``ti.ui.Window.get_scene()``; set its camera
        before calling this method and present it on the canvas afterward.
        Particle positions and optional colors are 1D three-component Taichi
        fields. ``particle_count`` selects the active prefix of a fixed solver
        buffer; zero draws no particles and None draws the whole field.
        """
        if particle_positions is not None:
            if not isfinite(particle_radius) or particle_radius <= 0:
                raise ValueError("particle_radius must be positive and finite")
            if particle_count is not None and (
                isinstance(particle_count, bool)
                or not isinstance(particle_count, Integral)
                or not 0 <= particle_count <= particle_positions.shape[0]
            ):
                raise ValueError("particle_count must be within the position field capacity")
        elif particle_count is not None or particle_colors is not None:
            raise ValueError("particle_count and particle_colors require particle_positions")

        lower, upper = self.world.domain_min, self.world.domain_max
        width, height, depth = (hi - lo for lo, hi in zip(lower, upper))
        center_x = (lower[0] + upper[0]) / 2
        scene.ambient_light(AMBIENT_LIGHT)
        scene.point_light(
            pos=(lower[0] - width * 0.25, upper[1] + height, lower[2] - depth * 0.5),
            color=(0.85, 0.80, 0.70),
        )
        scene.point_light(
            pos=(center_x + width, upper[1] + height * 0.5, upper[2] + depth),
            color=(0.35, 0.45, 0.60),
        )
        scene.mesh(
            self.vertices, normals=self.normals, per_vertex_color=self.colors,
            two_sided=True, vertex_count=self.vertex_count,
        )
        scene.lines(self.outline_vertices, width=2.0, color=OUTLINE_COLOR)
        if particle_positions is not None and particle_count != 0:
            scene.particles(
                particle_positions, radius=particle_radius, color=WATER_COLOR,
                per_vertex_color=particle_colors, index_count=particle_count,
            )
