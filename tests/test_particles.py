"""Lattice acceptance checks and real Taichi kernel reset tests.

The default backend is CPU for headless CI. Set FLUIDS_TEST_PARTICLE_ARCH=vulkan
to run the same field and kernel checks on Vulkan, with fallback disabled.
Particle downloads below are test inspection only.
"""

import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import taichi as ti

from fluids.config import PARTICLE_SPACING, REST_DENSITY, SETTINGS_SPECS, Settings
from fluids.particles import (
    COASTLINE_GAP, PARTICLE_CAPACITY, WATER_COLOR, ParticleState, plan_water_lattice,
)
from fluids.rendering import Renderer
from fluids.world import generate_world


class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.world = generate_world(self.settings)

    def test_deeper_water_increases_count_at_every_quality(self):
        for quality in PARTICLE_SPACING:
            with self.subTest(quality=quality):
                counts = [
                    plan_water_lattice(self.settings.with_updates(quality=quality, water_depth=depth), self.world).count
                    for depth in (0.5, 1.0, 2.0, 4.0)
                ]
                self.assertTrue(all(a < b for a, b in zip(counts, counts[1:])))

    def test_quality_increases_count_and_preserves_water_volume(self):
        for depth in (0.5, 0.7, 2.0, 4.0):
            lattices = [
                plan_water_lattice(self.settings.with_updates(quality=quality, water_depth=depth), self.world)
                for quality in PARTICLE_SPACING
            ]
            with self.subTest(depth=depth):
                self.assertTrue(all(a.count < b.count for a, b in zip(lattices, lattices[1:])))
                expected_volume = (20.0 - COASTLINE_GAP - 0.05) * (24.0 - 0.1) * depth
                for lattice in lattices:
                    self.assertAlmostEqual(lattice.count * lattice.mass / REST_DENSITY, expected_volume)
                    self.assertEqual(lattice.smoothing_radius, 2 * lattice.spacing)
                    self.assertEqual(lattice.collision_radius, lattice.spacing / 2)

    def test_regular_3d_lattice_and_collision_spheres_fit_ocean(self):
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 2.0, 4.0):
                settings = self.settings.with_updates(quality=quality, water_depth=depth)
                lattice = plan_water_lattice(settings, self.world)
                positions = lattice.positions()
                with self.subTest(quality=quality, depth=depth):
                    self.assertEqual(positions.shape, (lattice.count, 3))
                    self.assertEqual(positions.dtype, np.float32)
                    self.assertTrue(positions.flags.c_contiguous)
                    self.assertTrue(np.isfinite(positions).all())
                    radius = lattice.collision_radius
                    self.assertTrue(np.all(positions - radius >= np.array(self.world.domain_min) - 1e-6))
                    self.assertTrue(np.all(positions + radius <= np.array(self.world.domain_max) + 1e-6))
                    self.assertLessEqual(float(positions[:, 0].max()) + radius, -COASTLINE_GAP + 1e-6)
                    self.assertLessEqual(float(positions[:, 1].max()) + radius, depth + 1e-6)
                    for box in self.world.boxes:
                        inside = np.all((positions >= box.minimum) & (positions <= box.maximum), axis=1)
                        self.assertFalse(inside.any())
                    for axis, size in enumerate(lattice.shape):
                        coordinates = np.unique(positions[:, axis])
                        self.assertEqual(coordinates.size, size)
                        if size > 1:
                            np.testing.assert_allclose(np.diff(coordinates), settings.particle_spacing, atol=2e-6)

    def test_all_supported_depths_fit_capacity(self):
        # Count is monotone in depth. Checking the inclusive maximum proves the
        # bound for each quality; sample the intervening values as well.
        for quality in PARTICLE_SPACING:
            counts = []
            for depth in np.linspace(SETTINGS_SPECS["water_depth"].minimum, SETTINGS_SPECS["water_depth"].maximum, 71):
                lattice = plan_water_lattice(self.settings.with_updates(quality=quality, water_depth=depth), self.world)
                counts.append(lattice.count)
            self.assertLessEqual(max(counts), PARTICLE_CAPACITY)
            self.assertEqual(counts, sorted(counts))
        highest = plan_water_lattice(self.settings.with_updates(quality="high", water_depth=4.0), self.world)
        self.assertEqual(highest.count, 114_000)

    def test_city_controls_do_not_change_water(self):
        expected = plan_water_lattice(self.settings, self.world)
        for density in (0.0, 1.0):
            settings = self.settings.with_updates(city_density=density, city_seed=2**32 - 1,
                                                 building_min_height=0.5, building_max_height=10.0)
            self.assertEqual(plan_water_lattice(settings, generate_world(settings)), expected)

    def test_constructor_rejects_oversized_state_before_allocation(self):
        with patch.object(ti.Vector, "field") as allocate, patch.object(ti, "field") as scalar:
            with self.assertRaisesRegex(ValueError, "capacity"):
                ParticleState(self.settings, self.world, capacity=1)
        allocate.assert_not_called()
        scalar.assert_not_called()

    def test_invalid_capacities_and_ocean_dimensions(self):
        for capacity in (0, -1, True, 1.5, 2**31):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                plan_water_lattice(self.settings, self.world, capacity=capacity)
        too_deep = Settings(water_depth=4.0)
        shallow_world = replace(self.world, domain_max=(20.0, 3.0, 24.0))
        with self.assertRaisesRegex(ValueError, "ceiling"):
            plan_water_lattice(too_deep, shallow_world)


class ParticleFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arch = os.environ.get("FLUIDS_TEST_PARTICLE_ARCH", "cpu")
        if arch not in ("cpu", "vulkan"):
            raise ValueError("FLUIDS_TEST_PARTICLE_ARCH must be cpu or vulkan")
        ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def setUp(self):
        self.settings = Settings(quality="low", water_depth=0.5)
        self.world = generate_world(self.settings)
        count = plan_water_lattice(self.settings, self.world).count
        self.state = ParticleState(self.settings, self.world, capacity=2 * count + 10)

    def fields(self):
        return (self.state.positions, self.state.velocities, self.state.accelerations,
                self.state.densities, self.state.pressures, self.state.colors)

    def test_fields_are_bounded_and_count_is_separate(self):
        for field in self.fields():
            self.assertEqual(field.shape, (self.state.capacity,))
            self.assertEqual(field.dtype, ti.f32)
        self.assertLess(self.state.active_count, self.state.capacity)
        self.assertEqual(self.state.count[None], self.state.active_count)

    def test_reset_restores_layout_and_clears_all_dynamic_state(self):
        expected = self.state.positions.to_numpy()
        for field in self.fields():
            field.fill(99)
        self.state.count[None] = 0
        self.state.reset()
        np.testing.assert_array_equal(self.state.positions.to_numpy(), expected)
        for field in (self.state.velocities, self.state.accelerations, self.state.densities, self.state.pressures):
            np.testing.assert_array_equal(field.to_numpy(), 0)
        colors = self.state.colors.to_numpy()
        np.testing.assert_allclose(colors[:self.state.active_count], np.tile(WATER_COLOR, (self.state.active_count, 1)))
        np.testing.assert_array_equal(colors[self.state.active_count:], 0)
        self.assertEqual(self.state.count[None], self.state.active_count)

    def test_growing_then_shrinking_clears_old_particles_without_reallocation(self):
        original_fields = self.fields()
        original_count = self.state.active_count
        self.state.reset(self.settings.with_updates(water_depth=1.0))
        self.assertEqual(self.state.active_count, 2 * original_count)
        # Compiled kernels must use the new active count on every reset.
        with patch.object(ti.Vector, "field") as allocate, patch.object(ti, "field") as scalar:
            for _ in range(3):
                self.state.reset(self.settings)
                self.state.reset(self.settings.with_updates(water_depth=1.0))
            self.state.reset(self.settings)
        allocate.assert_not_called()
        scalar.assert_not_called()
        self.assertTrue(all(a is b for a, b in zip(original_fields, self.fields())))
        self.assertEqual(self.state.active_count, original_count)
        self.assertEqual(self.state.count[None], original_count)
        for field in self.fields():
            np.testing.assert_array_equal(field.to_numpy()[original_count:], 0)

    def test_oversized_reset_leaves_gpu_and_metadata_unchanged(self):
        self.state.velocities.fill(5)
        before = [field.to_numpy() for field in self.fields()]
        lattice = self.state.lattice
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.state.reset(self.settings.with_updates(quality="high", water_depth=4.0))
        for field, expected in zip(self.fields(), before):
            np.testing.assert_array_equal(field.to_numpy(), expected)
        self.assertIs(self.state.lattice, lattice)
        self.assertEqual(self.state.settings, self.settings)
        self.assertEqual(self.state.count[None], lattice.count)
        self.assertEqual(self.state.active_count, lattice.count)

    def test_supported_extremes_upload_and_quality_updates_parameters(self):
        state = ParticleState(self.settings, self.world)
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 4.0):
                settings = self.settings.with_updates(quality=quality, water_depth=depth)
                with self.subTest(quality=quality, depth=depth):
                    state.reset(settings)
                    lattice = plan_water_lattice(settings, self.world)
                    self.assertEqual(state.active_count, lattice.count)
                    self.assertEqual(state.count[None], lattice.count)
                    self.assertEqual(state.spacing, settings.particle_spacing)
                    self.assertEqual(state.smoothing_radius, 2 * settings.particle_spacing)
                    self.assertEqual(state.collision_radius, settings.particle_spacing / 2)
                    self.assertAlmostEqual(state.mass * state.active_count, REST_DENSITY * lattice.volume)
                    positions = state.positions.to_numpy()
                    np.testing.assert_array_equal(positions[:state.active_count], lattice.positions())
                    np.testing.assert_array_equal(positions[state.active_count:], 0)

    def test_reset_and_render_need_no_particle_downloads(self):
        renderer = Renderer(self.world)
        scene = Mock()
        with patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("particle download")):
            self.state.reset()
            renderer.render(scene, particle_positions=self.state.positions,
                            particle_colors=self.state.colors, particle_count=self.state.active_count,
                            particle_radius=self.state.collision_radius)
        scene.particles.assert_called_once_with(
            self.state.positions, radius=self.state.collision_radius, color=WATER_COLOR,
            per_vertex_color=self.state.colors, index_count=self.state.active_count,
        )


if __name__ == "__main__":
    unittest.main()
