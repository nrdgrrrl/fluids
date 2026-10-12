"""Device-staging contracts on CPU; opt in to Vulkan with the particle arch flag.

Real kernels and real FieldInfo/DeviceAllocation objects, with only the native
scene mocked. Downloads here inspect test results; they are never frame work.
These tests do not constitute a rendered comparison or GPU FPS measurement.
"""

import os
import unittest
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import taichi as ti
from taichi._lib import core as ti_core
from taichi.ui.scene import SceneV2
from taichi.ui.utils import get_field_info

from fluids.config import Settings
from fluids.rendering import (
    AMBIENT_LIGHT, OUTLINE_COLOR, OUTLINE_VERTEX_COUNT, VBO_STRIDE, WATER_COLOR,
    Renderer, _container_outline, _device_allocation, _device_field_info, _pack_vbo, _static_mesh,
)
from fluids.simulation import Simulation
from fluids.world import generate_world


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arch = os.environ.get("FLUIDS_TEST_PARTICLE_ARCH", "cpu")
        if arch not in ("cpu", "vulkan"):
            raise ValueError("FLUIDS_TEST_PARTICLE_ARCH must be cpu or vulkan")
        ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def setUp(self):
        self.world = generate_world(Settings())
        self.renderer = Renderer(self.world)
        self.core_scene = Mock(spec=ti_core.PySceneV2)
        self.scene = SceneV2(self.core_scene)
        self.positions = ti.Vector.field(3, ti.f32, shape=32)
        self.colors = ti.Vector.field(3, ti.f32, shape=32)
        self.positions.fill(7)
        self.colors.fill(0.4)

    def render(self, count=5, **kwargs):
        self.renderer.render(self.scene, particle_positions=self.positions,
                             particle_count=count, **kwargs)

    def assert_device_info(self, info, vbo, count):
        self.assertIsInstance(info, ti_core.FieldInfo)
        self.assertTrue(info.valid)
        self.assertEqual(info.field_source, ti_core.FieldSource.TaichiNDarray)
        self.assertNotEqual(info.field_source, ti_core.FieldSource.HostMappedPtr)
        self.assertEqual(info.dtype, ti.f32)
        self.assertEqual(info.shape, [count, VBO_STRIDE])
        self.assertEqual(info.num_elements, count * VBO_STRIDE)
        self.assertEqual(info.dev_alloc.alloc_id, _device_allocation(vbo.arr).alloc_id)

    def test_pinned_core_adapter_and_both_accessor_spellings(self):
        vbo = self.renderer._mesh_vbo
        info = _device_field_info(vbo, 4)
        self.assert_device_info(info, vbo, 4)
        allocation = info.dev_alloc
        for name in ("get_device_allocation", "device_allocation"):
            accessor = Mock(return_value=allocation)
            self.assertIs(_device_allocation(SimpleNamespace(**{name: accessor})), allocation)
            accessor.assert_called_once_with()
        preferred = Mock(return_value=allocation)
        fallback = Mock(side_effect=AssertionError("unneeded fallback"))
        _device_allocation(SimpleNamespace(get_device_allocation=preferred, device_allocation=fallback))
        preferred.assert_called_once_with()
        fallback.assert_not_called()
        for core in (SimpleNamespace(), SimpleNamespace(device_allocation=lambda: None)):
            with self.assertRaisesRegex(RuntimeError, "[Aa]llocation"):
                _device_allocation(core)
        # Exercise the installed wheel/core combination and show why this shim
        # is needed. Do not change the upstream helper or any class globally.
        if not hasattr(vbo.arr, "get_device_allocation"):
            with self.assertRaises(AttributeError):
                get_field_info(vbo)
        self.assertIs(import_module("taichi.ui.utils").get_field_info, get_field_info)

    def test_adapter_rejects_host_data_wrong_layout_and_bad_bounds(self):
        for data in (np.zeros((32, 12), np.float32), self.positions,
                     ti.Vector.ndarray(3, ti.f32, shape=32), ti.ndarray(ti.f32, shape=(32, 12)),
                     ti.Vector.ndarray(12, ti.f64, shape=32)):
            with self.subTest(data=type(data)), self.assertRaises(ValueError):
                _device_field_info(data, 4)
        with patch.object(self.renderer._mesh_vbo, "layout", ti.Layout.SOA):
            with self.assertRaises(ValueError):
                _device_field_info(self.renderer._mesh_vbo, 4)
        for count in (-1, 0, True, 1.5, 33):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _device_field_info(ti.Vector.ndarray(12, ti.f32, shape=32), count)

    def test_static_attributes_core_arguments_lights_and_camera(self):
        camera = SimpleNamespace(ptr=object())
        self.scene.set_camera(camera)
        self.renderer.render(self.scene)
        mesh_args = self.core_scene.mesh.call_args.args
        self.assert_device_info(mesh_args[0], self.renderer._mesh_vbo, self.renderer.vertex_count)
        self.assertEqual(mesh_args[1:], (True, self.renderer._no_indices, (0.5, 0.5, 0.5),
                                        True, self.renderer.vertex_count, 0,
                                        self.renderer.vertex_count, 0, False))
        self.assertFalse(mesh_args[2].valid)
        line_args = self.core_scene.lines.call_args.args
        self.assert_device_info(line_args[0], self.renderer._outline_vbo, OUTLINE_VERTEX_COUNT)
        self.assertEqual(line_args[1:], (self.renderer._no_indices, False, OUTLINE_COLOR, 2.0,
                                        OUTLINE_VERTEX_COUNT, 0, OUTLINE_VERTEX_COUNT, 0))
        self.core_scene.set_camera.assert_called_once_with(camera.ptr)
        self.core_scene.ambient_light.assert_called_once_with(AMBIENT_LIGHT)
        self.assertEqual(self.core_scene.point_light.call_args_list[0].args,
                         ((-30.0, 24.0, -12.0), (0.85, 0.80, 0.70)))
        self.assertEqual(self.core_scene.point_light.call_args_list[1].args,
                         ((40.0, 18.0, 48.0), (0.35, 0.45, 0.60)))
        self.core_scene.particles.assert_not_called()
        vertices, colors, normals, count = _static_mesh(self.world)
        packed = self.renderer._mesh_vbo.to_numpy()[:count]
        np.testing.assert_array_equal(packed[:, :3], vertices[:count])
        np.testing.assert_array_equal(packed[:, 3:6], normals[:count])
        np.testing.assert_array_equal(packed[:, 6:8], 0)
        np.testing.assert_array_equal(packed[:, 8:11], colors[:count])
        np.testing.assert_array_equal(packed[:, 11], 1)
        self.assertTrue(np.isfinite(packed).all())
        np.testing.assert_allclose(np.linalg.norm(packed[:, 3:6], axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(self.renderer._outline_vbo.to_numpy()[:, :3],
                                      _container_outline(self.world))

    def test_steady_frames_bypass_host_staging_and_pack_only_prefix(self):
        self.render(particle_colors=self.colors)
        vbo = self.renderer._particle_vbo
        vbo.fill(-77)
        before = self.renderer.staging_info()
        with (
            patch("taichi.ui.scene.get_vbo_field_v2", side_effect=AssertionError("host VBO")),
            patch("taichi.ui.scene.get_field_info", side_effect=AssertionError("stock metadata")),
            patch.object(type(self.positions), "to_numpy", side_effect=AssertionError("field download")),
            patch.object(type(vbo), "to_numpy", side_effect=AssertionError("VBO download")),
            patch.object(ti.Vector, "ndarray", side_effect=AssertionError("reallocation")),
            patch.object(self.renderer, "update_world", side_effect=AssertionError("static update")),
            patch("fluids.rendering._pack_vbo", wraps=_pack_vbo) as pack,
        ):
            for _ in range(3):
                self.render(particle_colors=self.colors)
        self.assertEqual(pack.call_count, 3)
        for call in pack.call_args_list:
            self.assertEqual(call.args, (vbo, self.positions, 0, self.colors, 5))
        packed = vbo.to_numpy()
        np.testing.assert_array_equal(packed[:5, :3], 7)
        np.testing.assert_array_equal(packed[:5, 3:8], 0)
        np.testing.assert_allclose(packed[:5, 8:11], 0.4)
        np.testing.assert_array_equal(packed[:5, 11], 1)
        np.testing.assert_array_equal(packed[5:], -77)
        info, has_colors, has_radii, color, radius, count, offset = self.core_scene.particles.call_args.args
        self.assert_device_info(info, vbo, 5)
        self.assertEqual((has_colors, has_radii, color, radius, count, offset),
                         (True, False, WATER_COLOR, 0.12, 5, 0))
        after = self.renderer.staging_info()
        self.assertEqual(after["particle_bytes_submitted"], 5 * 48)
        for name in ("particle_allocations", "mesh_packs", "outline_packs"):
            self.assertEqual(before[name], after[name])

    def test_count_zero_none_numpy_integer_and_color_changes(self):
        self.render(0)
        self.assertIsNone(self.renderer._particle_vbo)
        self.core_scene.particles.assert_not_called()
        self.render(None, particle_colors=self.colors)
        self.assert_device_info(self.renderer._particle_info, self.renderer._particle_vbo, 32)
        self.render(np.int64(3))
        self.assert_device_info(self.renderer._particle_info, self.renderer._particle_vbo, 3)
        packed = self.renderer._particle_vbo.to_numpy()[:3]
        np.testing.assert_array_equal(packed[:, 8:11], 0)
        np.testing.assert_array_equal(packed[:, 11], 1)
        self.assertFalse(self.core_scene.particles.call_args.args[1])
        rgba = ti.Vector.field(4, ti.f32, shape=3)
        rgba.fill(0.25)
        self.render(3, particle_colors=rgba)
        np.testing.assert_array_equal(self.renderer._particle_vbo.to_numpy()[:3, 8:12], 0.25)
        packs = self.renderer._particle_packs
        self.render(0)
        self.assertEqual(self.renderer._particle_packs, packs)
        self.assertEqual(self.renderer.staging_info()["particle_bytes_submitted"], 0)

    def test_invalid_particles_fail_before_any_scene_submission(self):
        cases = [{"particle_count": count} for count in (-1, 33, True, 1.5)]
        cases += [{"particle_radius": radius} for radius in (0, -1, float("nan"), float("inf"))]
        cases += [{"particle_positions": np.zeros((32, 3), np.float32)},
                  {"particle_positions": ti.Vector.field(2, ti.f32, shape=32)},
                  {"particle_positions": ti.Vector.field(3, ti.f32, shape=(4, 8))},
                  {"particle_colors": ti.Vector.field(3, ti.f32, shape=2)},
                  {"particle_colors": ti.field(ti.f32, shape=32)},
                  {"particle_positions": None, "particle_colors": self.colors}]
        for case in cases:
            args = {"particle_positions": self.positions, "particle_count": 5, **case}
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.renderer.render(self.scene, **args)
        self.assertEqual(self.core_scene.mock_calls, [])

    def test_new_source_fields_reuse_capacity_and_growth_synchronizes(self):
        other = ti.Vector.field(3, ti.f32, shape=32)
        larger = ti.Vector.field(3, ti.f32, shape=64)
        other.fill(9)
        larger.fill(11)
        self.render()
        original = self.renderer._particle_vbo
        self.renderer.render(self.scene, particle_positions=other, particle_count=4)
        self.assertIs(self.renderer._particle_vbo, original)
        np.testing.assert_array_equal(original.to_numpy()[:4, :3], 9)
        with patch.object(ti, "sync", wraps=ti.sync) as sync:
            self.renderer.render(self.scene, particle_positions=larger, particle_count=6)
        sync.assert_called_once_with()
        self.assertIsNot(self.renderer._particle_vbo, original)
        self.assertEqual(self.renderer._particle_allocations, 2)
        self.render(3)
        self.assertEqual(self.renderer._particle_vbo.shape, (64,))
        self.assert_device_info(self.renderer._particle_info, self.renderer._particle_vbo, 3)

    def test_ten_water_city_resets_reuse_buffers_and_skip_equal_geometry(self):
        settings = Settings(quality="low", water_depth=0.5)
        sim = Simulation(settings, capacity=5000)
        renderer = self.renderer

        def draw():
            renderer.render(self.scene, particle_positions=sim.particles.positions,
                            particle_count=sim.particles.active_count, particle_colors=sim.particles.colors)

        renderer.update_world(sim.world)
        draw()
        buffers = (renderer._mesh_vbo, renderer._outline_vbo, renderer._particle_vbo)
        original_fields = (renderer.vertices, renderer.colors, renderer.normals, renderer.outline_vertices)
        sim.paused = True
        camera = SimpleNamespace(ptr=object())
        self.scene.set_camera(camera)
        with (
            patch.object(ti.Vector, "ndarray", side_effect=AssertionError("new ndarray")),
            patch.object(ti.Vector, "field", side_effect=AssertionError("new field")),
            patch.object(ti, "field", side_effect=AssertionError("new scalar field")),
        ):
            for index in range(10):
                sim.reset(settings.with_updates(city_seed=index, city_density=index % 2,
                                               water_depth=0.5 if index % 2 else 1.0))
                self.assertTrue(renderer.update_world(sim.world))
                self.assertFalse(renderer.update_world(generate_world(sim.applied_settings)))
                draw()
                self.assertTrue(sim.paused)
                self.assert_device_info(renderer._particle_info, renderer._particle_vbo,
                                        sim.particles.active_count)
                self.assertEqual(renderer._mesh_info.shape[0], renderer.vertex_count)
                self.assertEqual(renderer._mesh_info.num_elements, renderer.vertex_count * 12)
        self.assertEqual(buffers, (renderer._mesh_vbo, renderer._outline_vbo, renderer._particle_vbo))
        self.assertEqual(original_fields, (renderer.vertices, renderer.colors, renderer.normals,
                                           renderer.outline_vertices))
        self.assertEqual(renderer._outline_packs, 1)
        self.assertEqual(renderer._particle_allocations, 1)
        self.core_scene.set_camera.assert_called_once_with(camera.ptr)
        count = sim.particles.active_count
        np.testing.assert_array_equal(renderer._particle_vbo.to_numpy()[:count, :3],
                                      sim.particles.positions.to_numpy()[:count])
        np.testing.assert_array_equal(renderer._particle_vbo.to_numpy()[:count, 8:11],
                                      sim.particles.colors.to_numpy()[:count])
        changed = replace(sim.world, domain_max=(20.0, 14.0, 24.0))
        renderer.update_world(changed)
        self.assertEqual(renderer._outline_packs, 2)
        np.testing.assert_array_equal(renderer._outline_vbo.to_numpy()[:, :3], _container_outline(changed))

    def test_unknown_taichi_version_fails_explicitly_before_allocation(self):
        with patch.object(ti, "__version__", (1, 8, 0)), patch.object(ti.Vector, "field") as allocate:
            with self.assertRaisesRegex(RuntimeError, "pinned Taichi 1.7.4"):
                Renderer(self.world)
        allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
