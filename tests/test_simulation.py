"""Coordinator acceptance checks; particle readback is confined to tests."""

import os
import unittest
from math import sqrt
from unittest.mock import patch

import numpy as np
import taichi as ti

from fluids.config import CFL_FACTOR, MAX_FRAME_TIME, MAX_SUBSTEPS, MAX_TIME_STEP, Settings
from fluids.density import PRESSURE_STIFFNESS
from fluids.grid import CELL_CAPACITY
from fluids.simulation import Simulation, SpatialGridOverflowError


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arch = os.environ.get("FLUIDS_TEST_PARTICLE_ARCH", "cpu")
        if arch not in ("cpu", "vulkan"):
            raise ValueError("FLUIDS_TEST_PARTICLE_ARCH must be cpu or vulkan")
        ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False, debug=True)
        cls.settings = Settings(quality="low", water_depth=0.5, damping=0, viscosity=0)
        cls.sim = Simulation(cls.settings, capacity=40_000)

    @classmethod
    def tearDownClass(cls):
        del cls.sim
        ti.reset()

    def setUp(self):
        self.sim.reset(self.settings)
        self.sim.paused = False

    def upload(self, points, velocities=None):
        state = self.sim.particles
        positions = np.zeros((state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        speeds = np.zeros_like(positions)
        if velocities is not None:
            speeds[:len(points)] = velocities
        state.positions.from_numpy(positions)
        state.velocities.from_numpy(speeds)
        state.count[None] = len(points)

    def test_reset_reproduces_world_and_water_and_reuses_all_gpu_fields(self):
        settings = self.settings.with_updates(city_seed=73)
        self.sim.reset(settings)
        original_world = self.sim.world
        state = self.sim.particles
        fields = (state.positions, state.velocities, state.accelerations,
                  state.densities, state.pressures, state.colors, state.count,
                  self.sim.grid.cell_counts, self.sim.grid.particle_indices,
                  self.sim.grid.overflow_count, self.sim.grid._origin,
                  self.sim.grid._radius, self.sim.grid._ready,
                  self.sim.dynamics.collisions.building_minimum,
                  self.sim.dynamics.collisions.building_maximum,
                  self.sim.dynamics.collisions.building_occupied,
                  self.sim.dynamics._step_limit)
        before = [field.to_numpy() for field in fields[:7]]
        self.upload([[-10, 5, 10]], [[50, 0, 0]])
        self.sim.advance(10)
        self.assertGreater(self.sim.simulated_time, 0)
        self.assertGreater(self.sim.discarded_time, 0)
        with (
            patch.object(ti, "field", side_effect=AssertionError("new field")),
            patch.object(ti.Vector, "field", side_effect=AssertionError("new vector field")),
        ):
            for quality in ("high", "medium", "low"):
                self.sim.reset(settings.with_updates(quality=quality, city_density=0))
                self.assertFalse(self.sim.dynamics.collisions.building_occupied.to_numpy().any())
            self.sim.reset(settings)
            self.sim.reset()
        self.assertEqual(self.sim.world, original_world)
        for field, expected in zip(fields[:7], before):
            np.testing.assert_array_equal(field.to_numpy(), expected)
        self.assertIs(self.sim.world, state.world)
        self.assertIs(self.sim.world, self.sim.dynamics.collisions.world)
        self.assertEqual(self.sim.grid._ready[None], 0)
        self.assertEqual(self.sim.grid.cell_counts.to_numpy().sum(), 0)
        self.assertEqual(self.sim.simulated_time, 0)
        self.assertEqual(self.sim.discarded_time, 0)
        self.assertEqual(self.sim.accumulated_time, 0)
        self.assertIsNone(self.sim.last_error)
        with self.assertRaisesRegex(RuntimeError, "compute_forces"):
            self.sim.dynamics.integrate(0.001)

    def test_structural_settings_are_pending_until_reset_and_physics_is_live(self):
        state = self.sim.particles
        world, lattice = self.sim.world, state.lattice
        self.upload([[-10, 5, 10]])
        live = self.settings.with_updates(gravity=20, quality="high", city_seed=123,
                                         water_depth=1, collision_friction=0.7)
        self.assertTrue(self.sim.requires_reset(live))
        self.assertFalse(self.sim.requires_reset(self.settings.with_updates(gravity=20)))
        dt = self.sim.step(live)
        np.testing.assert_allclose(state.velocities[0], [0, -20 * dt, 0], rtol=2e-6)
        self.assertIs(self.sim.world, world)
        self.assertIs(state.lattice, lattice)
        self.sim.paused = True
        self.sim.reset(live)
        self.assertFalse(self.sim.requires_reset(live))
        self.assertEqual(state.spacing, live.particle_spacing)
        self.assertIs(self.sim.dynamics.collisions.world, self.sim.world)
        self.assertTrue(self.sim.paused)

    def test_capacity_failure_leaves_scene_and_time_unchanged(self):
        self.upload([[-10, 5, 10]])
        self.sim.step()
        world, elapsed = self.sim.world, self.sim.simulated_time
        positions = self.sim.particles.positions.to_numpy()
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.sim.reset(self.settings.with_updates(quality="high", water_depth=4))
        self.assertIs(self.sim.world, world)
        self.assertEqual(self.sim.applied_settings, self.settings)
        self.assertEqual(self.sim.simulated_time, elapsed)
        np.testing.assert_array_equal(self.sim.particles.positions.to_numpy(), positions)

    def test_passes_execute_once_in_order_and_download_only_scalars(self):
        self.upload([[-10, 5, 10]])
        order = []

        def record(name, method):
            def wrapper(*args, **kwargs):
                order.append(name)
                return method(*args, **kwargs)
            return wrapper

        solver = self.sim.dynamics
        with (
            patch.object(self.sim.grid, "build", new=record("grid", self.sim.grid.build)),
            patch.object(solver.density, "_compute", new=record("density", solver.density._compute)),
            patch.object(solver, "_compute_forces", new=record("forces", solver._compute_forces)),
            patch.object(solver, "_integrate", new=record("integration", solver._integrate)),
            patch.object(solver.collisions, "resolve", new=record("collisions", solver.collisions.resolve)),
            patch.object(type(self.sim.particles.positions), "to_numpy", side_effect=AssertionError("download")),
        ):
            self.sim.step()
        self.assertEqual(order, ["grid", "density", "forces", "integration", "collisions"])

    def test_frame_chunking_preserves_physics_steps_and_fractional_time(self):
        live = self.settings.with_updates(gravity=0, damping=1)
        results = []
        for frames in ([0.05], [0.001] * 50):
            self.sim.reset()
            self.upload([[-10, 5, 10]], [[2, 0, 0]])
            steps = []
            with patch.object(self.sim.dynamics, "integrate", wraps=self.sim.dynamics.integrate) as integrate:
                for frame in frames:
                    self.sim.advance(frame, live)
                steps = [call.args[0] for call in integrate.call_args_list]
            results.append((self.sim.particles.positions.to_numpy(), steps, self.sim.simulated_time))
            self.assertAlmostEqual(self.sim.simulated_time + self.sim.accumulated_time, 0.05)
            self.assertEqual(self.sim.discarded_time, 0)
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][2], results[1][2])
        self.assertTrue(all(dt == MAX_TIME_STEP for dt in results[0][1]))

    def test_pause_and_zero_speed_clear_backlog_and_single_step_ignores_both(self):
        self.upload([[-10, 5, 10]], [[1, 0, 0]])
        live = self.settings.with_updates(gravity=0)
        self.sim.advance(0.001, live)
        self.assertGreater(self.sim.accumulated_time, 0)
        self.sim.paused = True
        self.assertEqual(self.sim.accumulated_time, 0)
        zero_speed = live.with_updates(playback_speed=0)
        with patch.object(self.sim.dynamics, "compute_forces", side_effect=AssertionError("paused")):
            self.assertEqual(self.sim.advance(100, live), 0)
            self.sim.paused = False
            self.assertEqual(self.sim.advance(100, zero_speed), 0)
        self.sim.paused = True
        elapsed = self.sim.simulated_time
        dt = self.sim.step(zero_speed)
        self.assertEqual(self.sim.last_substeps, 1)
        self.assertEqual(self.sim.simulated_time, elapsed + dt)
        self.assertEqual(dt, MAX_TIME_STEP)
        self.sim.paused = False
        self.assertEqual(self.sim.advance(0.001, live), 0)
        self.assertEqual(self.sim.simulated_time, elapsed + dt)
        self.assertAlmostEqual(self.sim.accumulated_time, 0.001)

    def test_overload_is_bounded_and_all_discarded_time_is_accounted_for(self):
        self.upload([[-10, 5, 10]], [[50, 0, 0]])
        live = self.settings.with_updates(gravity=0, playback_speed=4)
        self.assertEqual(self.sim.advance(10, live), MAX_SUBSTEPS)
        self.assertEqual(self.sim.accumulated_time, 0)
        self.assertLess(self.sim.simulated_time, MAX_FRAME_TIME * live.playback_speed)
        self.assertAlmostEqual(self.sim.simulated_time + self.sim.discarded_time, 40)
        elapsed = self.sim.simulated_time
        self.assertEqual(self.sim.advance(0, live), 0)
        self.assertEqual(self.sim.simulated_time, elapsed)

    def test_overflow_raises_specific_error_and_skips_all_physics_then_recovers(self):
        self.upload(np.tile([-10, 5, 10], (CELL_CAPACITY + 7, 1)))
        state = self.sim.particles
        fields = (state.positions, state.velocities, state.accelerations,
                  state.densities, state.pressures, state.colors)
        before = [field.to_numpy() for field in fields]

        def unsafe_kernel(*args):
            self.fail("physics kernel ran with incomplete bins")

        with (
            patch.object(self.sim.dynamics.density, "_compute", new=unsafe_kernel),
            patch.object(self.sim.dynamics, "_compute_forces", new=unsafe_kernel),
            patch.object(self.sim.dynamics, "_integrate", new=unsafe_kernel),
            patch.object(self.sim.dynamics.collisions, "resolve", side_effect=AssertionError("collide")),
        ):
            with self.assertRaises(SpatialGridOverflowError) as raised:
                self.sim.advance(0.02)
        self.assertEqual(raised.exception.overflow_count, 7)
        self.assertIs(raised.exception, self.sim.last_error)
        self.assertEqual(self.sim.simulated_time, 0)
        self.assertEqual(self.sim.last_substeps, 0)
        self.assertEqual(self.sim.accumulated_time, 0)
        self.assertAlmostEqual(self.sim.discarded_time, 0.02)
        for field, expected in zip(fields, before):
            np.testing.assert_array_equal(field.to_numpy(), expected)
        self.upload([[-10, 5, 10]])
        self.assertGreater(self.sim.step(), 0)
        self.assertIsNone(self.sim.last_error)

    def test_acoustic_displacement_and_acceleration_limits_bound_timestep(self):
        for speed, gravity in ((0, 0), (40, 20), (1, 20)):
            with self.subTest(speed=speed, gravity=gravity):
                self.sim.reset()
                self.upload([[-10, 5, 10]], [[speed, 0, 0]])
                dt = self.sim.step(self.settings.with_updates(gravity=gravity))
                radius = self.sim.particles.smoothing_radius
                self.assertLessEqual(dt, CFL_FACTOR * radius / (sqrt(PRESSURE_STIFFNESS) + speed) + 1e-8)
                travel = CFL_FACTOR * self.sim.particles.spacing
                self.assertLessEqual(speed * dt + gravity * dt**2, travel + 1e-8)
                if gravity:
                    self.assertLessEqual(dt, sqrt(CFL_FACTOR * radius / gravity) + 1e-8)

    def test_invalid_frame_times_are_rejected_without_changing_timing(self):
        for value in (-1, True, "0.1", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.sim.advance(value)
        self.assertEqual(self.sim.simulated_time, 0)
        self.assertEqual(self.sim.discarded_time, 0)
        self.assertEqual(self.sim.accumulated_time, 0)


if __name__ == "__main__":
    unittest.main()
