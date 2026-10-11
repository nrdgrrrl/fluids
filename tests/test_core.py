"""Small scene-contract checks; no Taichi initialization or desktop required."""

import random
import unittest
from dataclasses import FrozenInstanceError, fields

import numpy as np

from fluids.config import (
    BOUNDARY_MARGIN, PARTICLE_SPACING, RESET_FIELDS, REST_DENSITY,
    SETTINGS_SPECS, Settings,
)
from fluids.particles import COASTLINE_GAP, PARTICLE_CAPACITY, plan_water_lattice
from fluids.world import CITY_COLUMNS, CITY_ROWS, STREET_WIDTH, generate_world


class SettingsTests(unittest.TestCase):
    def test_defaults_and_inclusive_ranges_match_schema(self):
        settings = Settings()
        self.assertEqual({field.name for field in fields(settings)}, set(SETTINGS_SPECS))
        for name, spec in SETTINGS_SPECS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(settings, name), spec.default)
                endpoints = spec.choices or (spec.minimum, spec.maximum)
                for value in endpoints:
                    changes = {name: value}
                    if name.startswith("building_"):
                        changes.update(building_min_height=value, building_max_height=value)
                    self.assertEqual(getattr(settings.with_updates(**changes), name), value)

    def test_invalid_values_are_rejected_without_mutating_settings(self):
        settings = Settings()
        signature = settings.reset_signature()
        for name, spec in SETTINGS_SPECS.items():
            values = (None, True, "invalid", float("nan"), float("inf"), -float("inf"))
            if not spec.choices:
                values += (spec.minimum - 1, spec.maximum + 1)
                if spec.integer:
                    values += (0.5,)
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings.with_updates(**{name: value})
        with self.assertRaises(ValueError):
            settings.with_updates(building_min_height=6, building_max_height=5)
        with self.assertRaises(FrozenInstanceError):
            settings.gravity = 0
        self.assertEqual(settings, Settings())
        self.assertEqual(settings.reset_signature(), signature)

    def test_only_structural_edits_change_reset_signature(self):
        settings = Settings()
        for name, spec in SETTINGS_SPECS.items():
            value = spec.choices[0] if spec.choices else spec.minimum
            if value == getattr(settings, name) or name == "building_max_height":
                value = spec.maximum
            edited = settings.with_updates(**{name: value})
            with self.subTest(name=name):
                self.assertNotEqual(value, getattr(settings, name))
                self.assertEqual(edited.reset_signature() != settings.reset_signature(),
                                 name in RESET_FIELDS)


class WorldTests(unittest.TestCase):
    def test_seed_reproduces_geometry_mesh_and_water_without_global_rng_changes(self):
        rng_state = random.getstate()
        for seed in (0, 73, 2**32 - 1):
            settings = Settings(city_seed=seed)
            first, second = generate_world(settings), generate_world(settings)
            with self.subTest(seed=seed):
                self.assertEqual(first, second)
                for a, b in zip(first.triangle_mesh(), second.triangle_mesh()):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(plan_water_lattice(settings, first).positions(),
                                              plan_water_lattice(settings, second).positions())
        self.assertEqual(random.getstate(), rng_state)
        self.assertNotEqual(generate_world(Settings(city_seed=0)), generate_world(Settings(city_seed=1)))

    def test_density_adds_unchanged_plots_and_extremes_are_empty_or_full(self):
        full = generate_world(Settings(city_density=1))
        self.assertEqual(len(full.buildings), CITY_COLUMNS * CITY_ROWS)
        self.assertEqual(len(generate_world(Settings(city_density=0)).buildings), 0)
        for density in (0.25, 0.5, 0.75):
            world = generate_world(Settings(city_density=density))
            for plot, building in world.buildings_by_plot.items():
                self.assertEqual(building, full.buildings_by_plot[plot])

    def test_bounds_streets_and_mesh_at_height_extremes(self):
        for height in (0.5, 10.0):
            settings = Settings(city_density=1, building_min_height=height, building_max_height=height)
            world = generate_world(settings)
            width, depth = world.plot_size
            for box in world.boxes:
                lo, hi = np.array(box.minimum), np.array(box.maximum)
                self.assertTrue(np.all(lo >= world.domain_min))
                self.assertTrue(np.all(hi <= world.domain_max))
                self.assertTrue(np.all(hi > lo))
                if box.plot is not None:
                    column, row = box.plot
                    self.assertGreaterEqual(lo[0], column * width + STREET_WIDTH / 2)
                    self.assertLessEqual(hi[0], (column + 1) * width - STREET_WIDTH / 2)
                    self.assertGreaterEqual(lo[2], row * depth + STREET_WIDTH / 2)
                    self.assertLessEqual(hi[2], (row + 1) * depth - STREET_WIDTH / 2)
                    self.assertEqual(lo[1], world.land.maximum[1])
                    self.assertEqual(hi[1], height)
            vertices, colors = world.triangle_mesh()
            self.assertEqual(vertices.shape, (36 * len(world.boxes), 3))
            self.assertEqual(colors.shape, vertices.shape)
            self.assertEqual(vertices.dtype, np.float32)
            self.assertTrue(vertices.flags.c_contiguous)
            self.assertTrue(np.isfinite(vertices).all())
            self.assertTrue(np.isfinite(colors).all())
            self.assertTrue(((colors >= 0) & (colors <= 1)).all())
            for index, box in enumerate(world.boxes):
                mesh = vertices[36 * index:36 * (index + 1)]
                np.testing.assert_allclose(mesh.min(axis=0), box.minimum, atol=1e-6)
                np.testing.assert_allclose(mesh.max(axis=0), box.maximum, atol=1e-6)
                triangles = mesh.reshape(-1, 3, 3)
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                center = (np.array(box.minimum) + box.maximum) / 2
                self.assertTrue((np.sum(normals * (triangles.mean(axis=1) - center), axis=1) > 0).all())

    def test_invalid_world_dimensions_are_rejected(self):
        for width in (0, 12, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                generate_world(Settings(), domain_width=width)
        for height in (0, 1, float("nan"), float("inf")):
            with self.subTest(height=height), self.assertRaises(ValueError):
                generate_world(Settings(), land_height=height)


class CapacityTests(unittest.TestCase):
    def test_supported_extremes_fit_and_water_spheres_stay_in_ocean(self):
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 4.0):
                settings = Settings(quality=quality, water_depth=depth)
                world = generate_world(settings)
                lattice = plan_water_lattice(settings, world)
                with self.subTest(quality=quality, depth=depth):
                    self.assertGreater(lattice.count, 0)
                    self.assertLessEqual(lattice.count, PARTICLE_CAPACITY)
                    # Capacity is inclusive and rejected before allocating positions.
                    self.assertEqual(plan_water_lattice(settings, world, capacity=lattice.count), lattice)
                    with self.assertRaisesRegex(ValueError, "capacity"):
                        plan_water_lattice(settings, world, capacity=lattice.count - 1)
                    points, radius = lattice.positions(), lattice.collision_radius
                    lower = np.array(world.domain_min) + (BOUNDARY_MARGIN, 0, BOUNDARY_MARGIN)
                    upper = np.array((-COASTLINE_GAP, depth, world.domain_max[2] - BOUNDARY_MARGIN))
                    self.assertTrue((points - radius >= lower - 2e-6).all())
                    self.assertTrue((points + radius <= upper + 2e-6).all())
                    self.assertAlmostEqual(lattice.count * lattice.mass, REST_DENSITY * lattice.volume)


if __name__ == "__main__":
    unittest.main()
