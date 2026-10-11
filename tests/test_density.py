"""Real density kernels; particle downloads and CPU references are test-only.

Set FLUIDS_TEST_PARTICLE_ARCH=vulkan to verify the production compute backend.
"""

import os
import unittest
from itertools import product
from math import pi
from unittest.mock import patch

import numpy as np
import taichi as ti

from fluids.config import REST_DENSITY, Settings
from fluids.density import MAX_PRESSURE, PRESSURE_STIFFNESS, DensitySolver
from fluids.grid import CELL_CAPACITY, CELL_SIZE, SpatialGrid
from fluids.particles import ParticleState


def reference_density(points, radius, mass):
    points = np.asarray(points, dtype=np.float64)
    distances_squared = np.sum((points[:, None] - points[None, :]) ** 2, axis=2)
    weights = np.maximum(1.0 - distances_squared / radius**2, 0.0) ** 3
    return mass * 315.0 / (64.0 * pi * radius**3) * weights.sum(axis=1)


class DensityTests(unittest.TestCase):
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
        self.settings = Settings(quality="low", water_depth=0.5)
        self.state = ParticleState(self.settings)
        self.grid = SpatialGrid(self.state)
        self.solver = DensitySolver(self.grid)

    def upload(self, points):
        points = np.asarray(points, dtype=np.float32)
        positions = np.zeros((self.state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        self.state.positions.from_numpy(positions)
        self.state.count[None] = len(points)
        return points

    def test_default_ocean_density_and_pressure_are_finite(self):
        self.state.reset(Settings())
        self.assertEqual(self.solver.compute(), 0)
        densities = self.state.densities.to_numpy()[:self.state.active_count]
        pressures = self.state.pressures.to_numpy()[:self.state.active_count]
        self.assertTrue(np.isfinite(densities).all())
        self.assertTrue((densities > 0).all())
        self.assertTrue(np.isfinite(pressures).all())
        self.assertTrue(((pressures >= 0) & (pressures <= MAX_PRESSURE)).all())

    def test_isolated_particle_has_exact_self_density_and_no_tension(self):
        # A domain corner also checks that edge cells are not counted repeatedly.
        self.upload([self.state.world.domain_min])
        self.assertEqual(self.solver.compute(), 0)
        expected = self.state.mass * 315.0 / (64.0 * pi * self.state.smoothing_radius**3)
        self.assertAlmostEqual(self.state.densities[0], expected, delta=expected * 2e-6)
        self.assertGreater(self.state.densities[0], 0)
        self.assertEqual(self.state.pressures[0], 0)

    def test_neighbors_match_three_dimensional_poly6_reference(self):
        origin = np.asarray(self.state.world.domain_min)
        rng = np.random.default_rng(7)
        points = self.upload(np.concatenate([
            # All eight cells around an interior corner, including diagonals.
            origin + 3 * CELL_SIZE + np.asarray(list(product((-0.15, 0.15), repeat=3))),
            origin + rng.uniform(0.0, 2.0, size=(35, 3)),
            # Exact kernel support boundary, plus a neighbor just outside it.
            origin + [[8.0, 0.2, 0.2], [9.0, 0.2, 0.2], [9.01, 0.2, 0.2]],
        ]))
        self.state.densities.fill(-1)
        self.state.pressures.fill(-1)
        self.assertEqual(self.solver.compute(), 0)
        expected = reference_density(points, self.state.smoothing_radius, self.state.mass)
        np.testing.assert_allclose(self.state.densities.to_numpy()[:len(points)], expected, rtol=3e-6)
        pressure = np.clip(PRESSURE_STIFFNESS * (expected - REST_DENSITY), 0, MAX_PRESSURE)
        np.testing.assert_allclose(self.state.pressures.to_numpy()[:len(points)], pressure, rtol=1e-5, atol=0.1)
        # Only the GPU active count is authoritative; inactive data is untouched.
        self.assertNotEqual(self.state.active_count, len(points))
        np.testing.assert_array_equal(self.state.densities.to_numpy()[len(points):], -1)
        np.testing.assert_array_equal(self.state.pressures.to_numpy()[len(points):], -1)

    def test_compression_raises_pressure_and_extreme_compression_saturates(self):
        origin = np.asarray(self.state.world.domain_min)
        self.upload(origin + np.asarray(list(product((0.2, 1.4), repeat=3))))
        self.assertEqual(self.solver.compute(), 0)
        np.testing.assert_array_equal(self.state.pressures.to_numpy()[:8], 0)
        self.upload(origin + np.asarray(list(product((0.2, 0.4), repeat=3))))
        self.assertEqual(self.solver.compute(), 0)
        pressures = self.state.pressures.to_numpy()[:8]
        self.assertTrue((pressures > 0).all())
        self.assertTrue((pressures < MAX_PRESSURE).all())
        self.upload(np.tile(origin + 0.2, (CELL_CAPACITY, 1)))
        self.assertEqual(self.solver.compute(), 0)
        densities = self.state.densities.to_numpy()[:CELL_CAPACITY]
        self.assertTrue((densities > 2 * REST_DENSITY).all())
        self.assertTrue(np.isfinite(densities).all())
        np.testing.assert_array_equal(self.state.pressures.to_numpy()[:CELL_CAPACITY], MAX_PRESSURE)

    def test_quality_reset_updates_mass_radius_and_neighbors_in_same_solver(self):
        for quality in ("low", "high", "medium", "low"):
            with self.subTest(quality=quality):
                self.state.reset(self.settings.with_updates(quality=quality))
                origin = np.asarray(self.state.world.domain_min)
                points = self.upload(origin + [[1.4, 0.2, 0.2], [2.0, 0.2, 0.2]])
                self.assertEqual(self.solver.compute(), 0)
                expected = reference_density(points, self.state.smoothing_radius, self.state.mass)
                np.testing.assert_allclose(self.state.densities.to_numpy()[:2], expected, rtol=3e-6)

    def test_overflow_skips_density_and_next_successful_rebuild_recovers(self):
        origin = np.asarray(self.state.world.domain_min)
        self.state.densities.fill(123)
        self.state.pressures.fill(456)
        self.upload(np.tile(origin + 0.2, (CELL_CAPACITY + 3, 1)))
        with patch.object(self.solver, "_compute", new=lambda *args: self.fail("unsafe kernel launch")):
            self.assertEqual(self.solver.compute(), 3)
        np.testing.assert_array_equal(self.state.densities.to_numpy(), 123)
        np.testing.assert_array_equal(self.state.pressures.to_numpy(), 456)
        self.upload([origin + 0.2])
        self.assertEqual(self.solver.compute(), 0)
        self.assertGreater(self.state.densities[0], 0)
        self.assertEqual(self.state.pressures[0], 0)
        self.state.count[None] = 0
        self.assertEqual(self.solver.compute(), 0)

    def test_compute_requires_no_particle_downloads(self):
        with patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("download")):
            self.assertEqual(self.solver.compute(), 0)


if __name__ == "__main__":
    unittest.main()
