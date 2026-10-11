"""Dynamics acceptance checks with real kernels and independent CPU references.

Set FLUIDS_TEST_PARTICLE_ARCH=vulkan to exercise the production GPU backend.
Particle readback is used only for assertions in this module.
"""

import os
import unittest
from math import exp, pi
from unittest.mock import patch

import numpy as np
import taichi as ti

from fluids.config import (
    CFL_FACTOR, MAX_ACCELERATION, MAX_FRAME_TIME, MAX_SPEED, MAX_SUBSTEPS,
    MAX_TIME_STEP, MIN_DENSITY, MIN_TIME_STEP, REST_DENSITY, Settings,
)
from fluids.density import PRESSURE_STIFFNESS
from fluids.dynamics import DynamicsSolver, SPLASH_COLOR
from fluids.grid import CELL_CAPACITY, SpatialGrid
from fluids.particles import WATER_COLOR, ParticleState


def reference_forces(points, velocities, densities, pressures, radius, mass, settings):
    result = np.tile([0.0, -settings.gravity, 0.0], (len(points), 1))
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            offset = np.asarray(points[i], dtype=np.float64) - points[j]
            distance = np.linalg.norm(offset)
            if distance >= radius:
                continue
            rho_i, rho_j = max(densities[i], MIN_DENSITY), max(densities[j], MIN_DENSITY)
            if distance > 1e-6:
                gradient = -45 / (pi * radius**6) * (radius - distance)**2 * offset / distance
                result[i] -= mass * (pressures[i] / rho_i**2 + pressures[j] / rho_j**2) * gradient
            laplacian = 45 / (pi * radius**6) * (radius - distance)
            result[i] += settings.viscosity * mass / rho_j * (velocities[j] - velocities[i]) * laplacian
        magnitude = np.linalg.norm(result[i])
        if magnitude > MAX_ACCELERATION:
            result[i] *= MAX_ACCELERATION / magnitude
    return result


class DynamicsTests(unittest.TestCase):
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
        self.settings = Settings(quality="low", water_depth=0.5, damping=0.0)
        self.state = ParticleState(self.settings, capacity=20_000)
        self.grid = SpatialGrid(self.state)
        self.solver = DynamicsSolver(self.grid)

    def upload(self, points, velocities=None):
        points = np.asarray(points, dtype=np.float32)
        positions = np.zeros((self.state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        self.state.positions.from_numpy(positions)
        speeds = np.zeros_like(positions)
        if velocities is not None:
            speeds[:len(points)] = velocities
        self.state.velocities.from_numpy(speeds)
        self.state.count[None] = len(points)
        return points, speeds[:len(points)].copy()

    def test_gravity_falls_semi_implicitly_and_live_zero_gravity_stops_acceleration(self):
        points, _ = self.upload([[-10, 5, 10]])
        dt = 0.005
        self.assertEqual(self.solver.step(dt), 0)
        expected_velocity = [0, -self.settings.gravity * dt, 0]
        np.testing.assert_allclose(self.state.velocities[0], expected_velocity, rtol=2e-6)
        np.testing.assert_allclose(self.state.positions[0], points[0] + np.asarray(expected_velocity) * dt, atol=5e-7)
        velocity = np.array(self.state.velocities[0])
        no_gravity = self.settings.with_updates(gravity=0.0)
        self.assertEqual(self.solver.step(dt, no_gravity), 0)
        np.testing.assert_array_equal(self.state.accelerations[0], 0)
        np.testing.assert_array_equal(self.state.velocities[0], velocity)

    def test_launch_invalidates_prepared_forces_and_higher_speed_advances_toward_city(self):
        displacements = []
        for speed in (2, 8):
            points, _ = self.upload([[-12, 4, 12]])
            settings = self.settings.with_updates(gravity=0, viscosity=0,
                                                 wave_speed=speed, wave_height=0)
            self.assertEqual(self.solver.compute_forces(settings), 0)
            self.solver.launch_wave(settings)
            with self.assertRaisesRegex(RuntimeError, "compute_forces"):
                self.solver.integrate(0.005, settings)
            self.assertEqual(self.solver.step(0.005, settings), 0)
            displacements.append(self.state.positions[0][0] - points[0, 0])
            np.testing.assert_array_equal(self.state.velocities[0], [speed, 0, 0])
            self.assertEqual(self.state.count[None], 1)
        self.assertGreater(displacements[0], 0)
        self.assertGreater(displacements[1], displacements[0])

    def test_forces_match_reference_without_mutating_the_shared_snapshot(self):
        points, velocities = self.upload(
            [[-18.35, 1.55, 1.55], [-18.05, 1.75, 1.75], [-18.25, 1.95, 1.6],
             [-18.8, 1.6, 1.6], [-17.4, 2.2, 2.2], [-19.95, 0.05, 0.05],
             [-18.3, 1.7, 1.6], [-18.2, 1.7, 1.8], [-18.1, 1.6, 1.8]],
            [[1, 2, 3], [-2, 3, 1], [4, -3, -1], [0, 1, 0], [-1, -2, 1], [0, 0, 0],
             [1, -1, 0], [-2, 1, 0], [1, 0, 2]],
        )
        settings = self.settings.with_updates(viscosity=0.3)
        colors = self.state.colors.to_numpy()
        self.assertEqual(self.solver.compute_forces(settings), 0)
        densities = self.state.densities.to_numpy()[:len(points)]
        pressures = self.state.pressures.to_numpy()[:len(points)]
        self.assertTrue((pressures > 0).any())
        expected = reference_forces(points, velocities, densities, pressures,
                                    self.state.smoothing_radius, self.state.mass, settings)
        np.testing.assert_allclose(self.state.accelerations.to_numpy()[:len(points)], expected, rtol=1e-5, atol=2e-4)
        np.testing.assert_array_equal(self.state.positions.to_numpy()[:len(points)], points)
        np.testing.assert_array_equal(self.state.velocities.to_numpy()[:len(points)], velocities)
        np.testing.assert_array_equal(self.state.colors.to_numpy(), colors)
        # Integration uses the complete acceleration snapshot for every particle.
        dt = min(0.002, self.solver.stable_time_step)
        self.solver.integrate(dt, settings)
        expected_velocity = velocities + expected * dt
        expected_position = points + expected_velocity * dt
        # The corner particle starts outside the radius-adjusted walls. The
        # integration contract now includes container projection and impacts.
        lower = np.asarray(self.state.world.domain_min) + self.state.collision_radius
        upper = np.asarray(self.state.world.domain_max) - self.state.collision_radius
        for i in range(len(points)):
            for axis in range(3):
                direction = 0
                if expected_position[i, axis] <= lower[axis]:
                    expected_position[i, axis] = lower[axis]
                    direction = 1
                elif expected_position[i, axis] >= upper[axis]:
                    expected_position[i, axis] = upper[axis]
                    direction = -1
                if expected_velocity[i, axis] * direction < 0:
                    normal_speed = expected_velocity[i, axis]
                    expected_velocity[i] *= 1 - settings.collision_friction
                    expected_velocity[i, axis] = -settings.collision_restitution * normal_speed
        np.testing.assert_allclose(self.state.velocities.to_numpy()[:len(points)], expected_velocity, rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(self.state.positions.to_numpy()[:len(points)], expected_position, atol=2e-6)

    def test_pressure_is_repulsive_and_symmetric_for_unequal_density(self):
        self.upload([[-10, 4, 10], [-9.7, 4, 10], [-9.55, 4.2, 10]])
        settings = self.settings.with_updates(gravity=0, viscosity=0)
        self.assertEqual(self.solver.compute_forces(settings), 0)
        # Exercise the density floor and unequal density pair explicitly.
        self.state.count[None] = 2
        self.state.densities[0], self.state.densities[1] = 0, 1200
        self.state.pressures[0], self.state.pressures[1] = 20, 200
        # Remove the third index from the grid before the independent pair pass.
        self.grid.build()
        self.solver._compute_forces(self.state.smoothing_radius, self.state.mass, 0, 0)
        acceleration = self.state.accelerations.to_numpy()[:2]
        self.assertLess(acceleration[0, 0], 0)
        np.testing.assert_allclose(acceleration[0], -acceleration[1], atol=2e-5)
        expected = reference_forces(np.asarray([[-10, 4, 10], [-9.7, 4, 10]], dtype=np.float32),
                                    np.zeros((2, 3)), [0, 1200], [20, 200],
                                    self.state.smoothing_radius, self.state.mass, settings)
        np.testing.assert_allclose(acceleration, expected, rtol=3e-6, atol=2e-5)

    def test_higher_viscosity_reduces_relative_motion_and_constant_velocity_is_unchanged(self):
        points = [[-10, 4, 10], [-9.6, 4, 10]]
        velocities = [[2, 0, 0], [-2, 0, 0]]
        differences = []
        for viscosity in (0.0, 0.1, 1.0):
            self.upload(points, velocities)
            settings = self.settings.with_updates(gravity=0, viscosity=viscosity)
            self.assertEqual(self.solver.step(0.005, settings), 0)
            speeds = self.state.velocities.to_numpy()[:2]
            differences.append(np.linalg.norm(speeds[0] - speeds[1]))
            np.testing.assert_allclose(speeds.sum(axis=0), 0, atol=1e-6)
        self.assertGreater(differences[0], differences[1])
        self.assertGreater(differences[1], differences[2])
        self.upload(points, [[2, 1, 0], [2, 1, 0]])
        self.assertEqual(self.solver.step(0.005, settings), 0)
        np.testing.assert_array_equal(self.state.accelerations.to_numpy()[:2], 0)

    def test_live_damping_uses_time_based_decay_and_preserves_particle_count(self):
        points = [[-10, 4, 10]]
        for damping in (0, 1, 10):
            self.upload(points, [[3, 1, 2]])
            settings = self.settings.with_updates(gravity=0, damping=damping)
            self.assertEqual(settings.reset_signature(), self.settings.reset_signature())
            self.assertEqual(self.solver.step(0.005, settings), 0)
            expected = np.asarray([3, 1, 2]) * exp(-damping * 0.005)
            np.testing.assert_allclose(self.state.velocities[0], expected, rtol=2e-6)
            self.assertEqual(self.state.count[None], 1)
            self.assertEqual(self.state.active_count, self.state.lattice.count)
        # The same elapsed time in two steps gives the same damping decay.
        self.upload(points, [[3, 1, 2]])
        self.solver.step(0.0025, settings)
        self.solver.step(0.0025, settings)
        np.testing.assert_allclose(self.state.velocities[0], expected, rtol=2e-6)

    def test_coincident_particles_and_extreme_motion_remain_bounded(self):
        self.upload([[-10, 4, 10], [-10, 4, 10]], [[100, 0, 0], [-100, 0, 0]])
        settings = self.settings.with_updates(gravity=20, viscosity=1)
        self.assertEqual(self.solver.compute_forces(settings), 0)
        acceleration = self.state.accelerations.to_numpy()[:2]
        self.assertTrue(np.isfinite(acceleration).all())
        self.assertTrue((np.linalg.norm(acceleration, axis=1) <= MAX_ACCELERATION + 1e-3).all())
        self.solver.integrate(self.solver.stable_time_step, settings)
        velocity = self.state.velocities.to_numpy()[:2]
        self.assertTrue(np.isfinite(velocity).all())
        self.assertTrue((np.linalg.norm(velocity, axis=1) <= MAX_SPEED + 1e-4).all())
        self.assertTrue(np.isfinite(self.state.positions.to_numpy()[:2]).all())

    def test_viscosity_limit_prevents_dense_cluster_velocity_reversal(self):
        count = 100
        velocities = np.tile([-1, 0, 0], (count, 1))
        velocities[0] = [1, 0, 0]
        self.upload(np.tile([-10, 4, 10], (count, 1)), velocities)
        settings = self.settings.with_updates(gravity=0, viscosity=1)
        self.assertEqual(self.solver.compute_forces(settings), 0)
        # Stress the safeguard at its documented density floor, even when a
        # dense neighborhood is paired with an artificially low density field.
        self.state.densities.fill(0)
        self.solver._compute_forces(self.state.smoothing_radius, self.state.mass, 1, 0)
        self.assertLess(self.solver.stable_time_step, MAX_TIME_STEP)
        with self.assertRaisesRegex(ValueError, "stability limit"):
            self.solver.integrate(MAX_TIME_STEP, settings)
        self.solver.integrate(self.solver.stable_time_step, settings)
        speeds = self.state.velocities.to_numpy()[:count]
        self.assertGreaterEqual(speeds[0, 0], speeds[1, 0])

    def test_splash_color_requires_both_speed_and_low_density_and_padding_is_untouched(self):
        self.upload([[-10, 4, 10], [-6, 4, 10], [-2, 4, 10]],
                    [[0, 0, 0], [5, 0, 0], [5, 0, 0]])
        settings = self.settings.with_updates(gravity=0, viscosity=0)
        self.assertEqual(self.solver.compute_forces(settings), 0)
        self.state.densities[2] = REST_DENSITY
        self.state.colors[3] = [0.2, 0.3, 0.4]
        self.state.accelerations[3] = [4, 5, 6]
        self.solver.integrate(0.005, settings)
        np.testing.assert_allclose(self.state.colors[0], WATER_COLOR, atol=1e-6)
        np.testing.assert_allclose(self.state.colors[2], WATER_COLOR, atol=1e-6)
        self.assertTrue(np.all(np.asarray(self.state.colors[1]) > WATER_COLOR))
        self.assertTrue(np.all(np.asarray(self.state.colors[1]) <= np.asarray(SPLASH_COLOR) + 1e-6))
        np.testing.assert_allclose(self.state.colors[3], [0.2, 0.3, 0.4], atol=1e-6)
        np.testing.assert_array_equal(self.state.accelerations[3], [4, 5, 6])
        np.testing.assert_array_equal(self.state.positions[3], 0)
        np.testing.assert_array_equal(self.state.velocities[3], 0)

    def test_overflow_skips_all_dynamics_and_recovers_after_rebuild(self):
        self.upload(np.tile([-10, 4, 10], (CELL_CAPACITY + 3, 1)))
        self.state.accelerations.fill(42)
        fields = (self.state.positions, self.state.velocities, self.state.accelerations,
                  self.state.densities, self.state.pressures, self.state.colors)
        before = [field.to_numpy() for field in fields]
        with patch.object(self.solver, "_compute_forces", side_effect=AssertionError("unsafe force pass")):
            self.assertEqual(self.solver.step(0.005), 3)
        for field, expected in zip(fields, before):
            np.testing.assert_array_equal(field.to_numpy(), expected)
        with self.assertRaises(RuntimeError):
            self.solver.integrate(0.005)
        self.upload([[-10, 4, 10]])
        self.assertEqual(self.solver.step(0.005), 0)

    def test_quality_reset_updates_compiled_force_parameters(self):
        settings = self.settings.with_updates(gravity=0, viscosity=0.2)
        for quality in ("low", "high", "medium", "low"):
            self.state.reset(settings.with_updates(quality=quality))
            points, velocities = self.upload([[-10, 4, 10], [-9.7, 4, 10]], [[1, 2, 0], [-1, 0, 1]])
            self.assertEqual(self.solver.compute_forces(), 0)
            expected = reference_forces(points, velocities,
                                        self.state.densities.to_numpy()[:2], self.state.pressures.to_numpy()[:2],
                                        self.state.smoothing_radius, self.state.mass, self.state.settings)
            np.testing.assert_allclose(self.state.accelerations.to_numpy()[:2], expected, rtol=5e-6, atol=1e-4)

    def test_frame_limits_playback_pause_and_no_particle_downloads(self):
        points, _ = self.upload([[-10, 4, 10]], [[1, 0, 0]])
        fields = (self.state.positions, self.state.velocities, self.state.accelerations,
                  self.state.densities, self.state.pressures, self.state.colors)
        before = [field.to_numpy() for field in fields]
        paused = self.settings.with_updates(playback_speed=0)
        with patch.object(self.solver, "compute_forces", side_effect=AssertionError("paused")):
            self.assertEqual(self.solver.advance(1, paused), 0)
        for field, expected in zip(fields, before):
            np.testing.assert_array_equal(field.to_numpy(), expected)
        live = self.settings.with_updates(gravity=0)
        with patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("download")):
            with patch.object(self.solver, "_integrate", wraps=self.solver._integrate) as integrate:
                self.assertEqual(self.solver.advance(10, live), 0)
        elapsed = sum(call.args[0] for call in integrate.call_args_list)
        self.assertLessEqual(integrate.call_count, MAX_SUBSTEPS)
        self.assertAlmostEqual(elapsed, MAX_FRAME_TIME)
        np.testing.assert_allclose(self.state.positions[0], points[0] + [elapsed, 0, 0], atol=1e-5)
        with patch.object(self.solver, "_integrate", wraps=self.solver._integrate) as integrate:
            self.assertEqual(self.solver.advance(10, live.with_updates(playback_speed=4)), 0)
        self.assertEqual(integrate.call_count, MAX_SUBSTEPS)
        self.assertTrue(all(MIN_TIME_STEP <= call.args[0] <= MAX_TIME_STEP for call in integrate.call_args_list))

    def test_cfl_limit_responds_to_speed_and_integration_requires_fresh_forces(self):
        self.upload([[-10, 4, 10]], [[40, 0, 0]])
        self.assertEqual(self.solver.compute_forces(), 0)
        expected = CFL_FACTOR * self.state.smoothing_radius / (PRESSURE_STIFFNESS**0.5 + 40)
        self.assertAlmostEqual(self.solver.stable_time_step, expected, delta=1e-8)
        self.solver.integrate(self.solver.stable_time_step)
        with self.assertRaises(RuntimeError):
            self.solver.integrate(MIN_TIME_STEP)
        self.state.count[None] = 0
        self.assertEqual(self.solver.step(MIN_TIME_STEP), 0)

    def test_invalid_times_and_damping_are_rejected(self):
        for value in (float("nan"), float("inf"), True, "0.01", -1, 0, MIN_TIME_STEP / 2, 2 * MAX_TIME_STEP):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.solver.step(value)
        for value in (float("nan"), float("inf"), True, "0.01", -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.solver.advance(value)
        for value in (-1, 11, float("nan"), True):
            with self.subTest(damping=value), self.assertRaises(ValueError):
                self.settings.with_updates(damping=value)


if __name__ == "__main__":
    unittest.main()
