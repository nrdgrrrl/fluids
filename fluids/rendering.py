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
from taichi._lib import core as ti_core
from taichi.lang._ndarray import Ndarray
from taichi.lang.matrix import MatrixField

from .world import World


# A full 6 x 8 city, land, and floor need only 590 triangles. Keep spare room
# without allocating new fields when density or the city seed changes.
STATIC_VERTEX_CAPACITY = 3 * 2048
OUTLINE_VERTEX_COUNT = 24  # two endpoints for each of the twelve box edges
FLOOR_COLOR = (0.12, 0.24, 0.31)
OUTLINE_COLOR = (0.72, 0.82, 0.90)
WATER_COLOR = (0.04, 0.48, 0.85)
AMBIENT_LIGHT = (0.45, 0.45, 0.50)
VBO_STRIDE = 12  # GGUI 1.7.4: position(3), normal(3), uv(2), RGBA(4)


def _device_allocation(array):
    """Handle the pinned wheel's Python/core ndarray naming mismatch locally."""
    for name in ("get_device_allocation", "device_allocation"):
        getter = getattr(array, name, None)
        if callable(getter):
            allocation = getter()
            if not isinstance(allocation, ti_core.DeviceAllocation):
                raise RuntimeError("Taichi ndarray accessor did not return a DeviceAllocation")
            return allocation
    raise RuntimeError("Taichi 1.7.4 core has no supported ndarray allocation accessor")


def _device_field_info(vbo, count):
    """Describe only a device VBO's populated prefix to the GGUI core.

    This replaces ui.utils.get_field_info's broken ndarray branch, without
    changing Taichi globals. The core copies num_elements * sizeof(float),
    independently of the draw count, so both metadata bounds matter.
    """
    if not isinstance(vbo, Ndarray) or vbo.dtype != ti.f32 or (
        len(vbo.shape) != 1 or vbo.element_shape != (VBO_STRIDE,) or vbo.layout != ti.Layout.AOS
    ):
        raise ValueError("GGUI VBO must be a 1D device ndarray of 12-component float32 vectors")
    if isinstance(count, bool) or not isinstance(count, Integral) or not 0 < count <= vbo.shape[0]:
        raise ValueError("GGUI VBO prefix must be positive and within capacity")
    info = ti_core.FieldInfo()
    info.valid = True
    info.field_source = ti_core.FieldSource.TaichiNDarray
    info.dev_alloc = _device_allocation(vbo.arr)
    info.dtype = vbo.dtype
    info.num_elements = int(count) * VBO_STRIDE
    info.shape = [int(count), VBO_STRIDE]
    return info


def _validate_vector_field(field, name, components):
    if not isinstance(field, MatrixField) or (
        len(field.shape) != 1 or field.m != 1
        or field.n not in components or field.dtype != ti.f32
    ):
        raise ValueError(f"{name} must be a 1D float32 Taichi vector field with {components} components")


@ti.kernel
def _pack_vbo(
    vbo: ti.types.ndarray(element_dim=1),
    positions: ti.template(), normals: ti.template(), colors: ti.template(), count: ti.i32,
):
    # Runtime count bounds work and writes, even when the solver reserves much
    # more space. Initialize every lane so optional attributes cannot go stale.
    for i in range(count):
        vbo[i] = ti.Vector.zero(ti.f32, VBO_STRIDE)
        vbo[i][0:3] = positions[i]
        vbo[i][11] = 1.0
        if ti.static(normals != 0):
            vbo[i][3:6] = normals[i]
        if ti.static(colors != 0):
            if ti.static(colors.n == 3):
                vbo[i][8:11] = colors[i]
            else:
                vbo[i][8:12] = colors[i]


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
    """Reusable fields and device VBOs shared by world resets and frames.

    Construct once after ``initialize_gpu()``. ``update_world`` performs CPU
    mesh generation, uploads and VBO packing only for changed geometry.
    ``render`` packs the active water prefix into a reusable device ndarray.
    It bypasses SceneV2's NumPy staging via the pinned 1.7.4 core interface;
    Vulkan copies the populated prefix between buffers on the same device.
    GGUI still owns internal render buffers (including unused host-writable
    staging allocations); this path does not map them to upload vertex data.
    The camera belongs to the caller so orbit controls remain independent.
    """

    def __init__(self, world: World) -> None:
        if ti.__version__ != (1, 7, 4):
            raise RuntimeError("Device GGUI staging requires the pinned Taichi 1.7.4 core API")
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.outline_vertices = ti.Vector.field(3, dtype=ti.f32, shape=OUTLINE_VERTEX_COUNT)
        self._mesh_vbo = ti.Vector.ndarray(VBO_STRIDE, ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self._outline_vbo = ti.Vector.ndarray(VBO_STRIDE, ti.f32, shape=OUTLINE_VERTEX_COUNT)
        self._particle_vbo = None
        self._particle_info = None
        self._no_indices = ti_core.FieldInfo()
        self._no_indices.valid = False
        self._mesh_packs = self._outline_packs = self._particle_packs = 0
        self._particle_allocations = 0
        self._last_particle_count = 0
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
        _pack_vbo(self._mesh_vbo, self.vertices, self.normals, self.colors, vertex_count)
        self._mesh_info = _device_field_info(self._mesh_vbo, vertex_count)
        self._mesh_packs += 1
        if self.world is None or (
            world.domain_min != self.world.domain_min
            or world.domain_max != self.world.domain_max
        ):
            self.outline_vertices.from_numpy(_container_outline(world))
            _pack_vbo(self._outline_vbo, self.outline_vertices, 0, 0, OUTLINE_VERTEX_COUNT)
            self._outline_info = _device_field_info(self._outline_vbo, OUTLINE_VERTEX_COUNT)
            self._outline_packs += 1
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
        Particle positions are 1D float32 three-component Taichi fields; colors
        may have three or four components. ``particle_count`` selects the active
        prefix of a fixed solver buffer; zero draws no particles and None draws
        the whole field.
        """
        if particle_positions is not None:
            _validate_vector_field(particle_positions, "particle_positions", (3,))
            if not isfinite(particle_radius) or particle_radius <= 0:
                raise ValueError("particle_radius must be positive and finite")
            if particle_count is not None and (
                isinstance(particle_count, bool)
                or not isinstance(particle_count, Integral)
                or not 0 <= particle_count <= particle_positions.shape[0]
            ):
                raise ValueError("particle_count must be within the position field capacity")
            particle_count = particle_positions.shape[0] if particle_count is None else int(particle_count)
            if particle_colors is not None:
                _validate_vector_field(particle_colors, "particle_colors", (3, 4))
                if particle_colors.shape[0] < particle_count:
                    raise ValueError("particle_colors must cover the active position prefix")
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
        self._draw_static(scene)
        self._last_particle_count = 0
        if particle_positions is not None and particle_count != 0:
            self._draw_particles(scene, particle_positions, particle_colors, particle_radius, particle_count)

    def _draw_static(self, scene):
        # Match SceneV2's positional core arguments, without its host VBO helper.
        scene.scene.mesh(
            self._mesh_info, True, self._no_indices, (0.5, 0.5, 0.5), True,
            self.vertex_count, 0, self.vertex_count, 0, False,
        )
        scene.scene.lines(
            self._outline_info, self._no_indices, False, OUTLINE_COLOR, 2.0,
            OUTLINE_VERTEX_COUNT, 0, OUTLINE_VERTEX_COUNT, 0,
        )

    def _draw_particles(self, scene, positions, colors, radius, count):
        if self._particle_vbo is None or self._particle_vbo.shape[0] < positions.shape[0]:
            # Ordinary water/city resets reuse solver capacity and never enter
            # this branch. Drain pending uses before replacing an allocation.
            if self._particle_vbo is not None:
                ti.sync()
            vbo = ti.Vector.ndarray(VBO_STRIDE, ti.f32, shape=positions.shape[0])
            info = _device_field_info(vbo, count)
            self._particle_vbo, self._particle_info = vbo, info
            self._particle_allocations += 1
        elif self._particle_info.shape[0] != count:
            self._particle_info = _device_field_info(self._particle_vbo, count)
        _pack_vbo(self._particle_vbo, positions, 0, colors if colors is not None else 0, count)
        self._particle_packs += 1
        self._last_particle_count = count
        scene.scene.particles(
            self._particle_info, colors is not None, False, WATER_COLOR, radius, count, 0,
        )

    def staging_info(self):
        """Metadata-only diagnostics; never download or map vertex contents."""
        capacity = 0 if self._particle_vbo is None else self._particle_vbo.shape[0]
        return {
            "source": str(self._mesh_info.field_source), "backend": str(ti.cfg.arch),
            "particle_source": None if self._particle_info is None else str(self._particle_info.field_source),
            "particle_capacity": capacity,
            "particle_allocations": self._particle_allocations,
            "particle_packs": self._particle_packs,
            "particles_packed_last_frame": self._last_particle_count,
            "particle_bytes_submitted": self._last_particle_count * VBO_STRIDE * 4,
            "device_vbo_bytes_reserved": (STATIC_VERTEX_CAPACITY + OUTLINE_VERTEX_COUNT + capacity) * VBO_STRIDE * 4,
            "mesh_packs": self._mesh_packs, "outline_packs": self._outline_packs,
        }
