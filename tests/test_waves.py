"""Wave acceptance checks using real Taichi kernels and CPU references.

Set FLUIDS_TEST_PARTICLE_ARCH=vulkan to exercise the production backend.
Particle downloads here are assertion-only, never part of the launcher.
"""

import os
import unittest
from math import sqrt
from unittest.mock import patch

import numpy as np
import taichi as ti

from fluids.config import IMMEDIATE_FIELDS, MAX_SPEED, SETTINGS_SPECS, Settings
from fluids.particles import ParticleState
from fluids.waves import IMPULSE_CENTER_X, REFERENCE_ACCELERATION, WaveLauncher


class WaveSettingsTests(unittest.TestCase):
    def test_launch_controls_are_validated_and_independent_of_water_quantity(self):
        settings = Settings()
        updated = settings.with_updates(wave_speed=20, wave_width=12, wave_height=10)
        self.assertEqual(updated.reset_signature(), settings.reset_signature())
        self.assertNotEqual(settings.with_updates(water_depth=4).reset_signature(),
                            settings.reset_signature())
        for name in ("wave_speed", "wave_width", "wave_height"):
            self.assertIn(name, IMMEDIATE_FIELDS)
            spec = SETTINGS_SPECS[name]
            for value in (spec.minimum, spec.maximum):
                settings.with_updates(**{name: value})
            for value in (spec.minimum - 0.01, spec.maximum + 0.01,
                          float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings.with_updates(**{name: value})


class WaveLauncherTests(unittest.TestCase):
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
        self.state = ParticleState(self.settings, capacity=4000)
        self.launcher = WaveLauncher(self.state)

    def upload(self, points, velocities=None):
        positions = np.zeros((self.state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        speeds = np.zeros_like(positions)
        if velocities is not None:
            speeds[:len(points)] = velocities
        self.state.positions.from_numpy(positions)
        self.state.velocities.from_numpy(speeds)
        self.state.count[None] = len(points)

    def test_cosine_impulse_only_changes_velocity_in_full_width_offshore_band(self):
        # Check both exact edges, both sides of the center, and multiple y/z
        # coordinates. Existing sideways motion must be retained below the cap.
        points = [[x, y, z] for x, y, z in (
            (-16, 1, 2), (-15, 2, 4), (-14, 1, 6), (-12, 1, 8),
            (-12, 3, 16), (-10, 2, 18), (-9, 1, 20), (-8, 1, 22), (2, 1, 12),
        )]
        velocities = np.tile([-1, -2, 3], (len(points), 1))
        self.upload(points, velocities)
        # An inactive slot inside the band must also remain untouched.
        self.state.positions[len(points)] = [-12, 1, 12]
        self.state.velocities[len(points)] = [4, 5, 6]
        for field in (self.state.accelerations, self.state.densities,
                      self.state.pressures, self.state.colors):
            field.fill(7)
        other_fields = (self.state.positions, self.state.accelerations,
                        self.state.densities, self.state.pressures, self.state.colors)
        before = [field.to_numpy() for field in other_fields]
        inactive_velocity = self.state.velocities.to_numpy()[len(points):].copy()
        metadata = (self.state.active_count, self.state.mass, self.state.lattice, self.state.settings)

        settings = self.settings.with_updates(wave_width=6, wave_speed=8, wave_height=2)
        self.launcher.launch(settings)
        offset = np.asarray(points)[:, 0] - IMPULSE_CENTER_X
        envelope = np.where(np.abs(offset) < 3, 0.5 * (1 + np.cos(2 * np.pi * offset / 6)), 0)
        expected = velocities + envelope[:, None] * [8, sqrt(2 * settings.gravity * 2), 0]
        actual = self.state.velocities.to_numpy()
        np.testing.assert_allclose(actual[:len(points)], expected, atol=2e-6)
        np.testing.assert_array_equal(actual[:len(points)][envelope == 0], velocities[envelope == 0])
        np.testing.assert_array_equal(actual[len(points):], inactive_velocity)
        for field, values in zip(other_fields, before):
            np.testing.assert_array_equal(field.to_numpy(), values)
        self.assertEqual(self.state.count[None], len(points))
        self.assertEqual(metadata, (self.state.active_count, self.state.mass,
                                    self.state.lattice, self.state.settings))

    def test_live_wider_launch_affects_more_existing_particles(self):
        self.upload([[-18, 1, 4], [-16, 1, 6], [-14, 1, 8],
                     [-12, 1, 10], [-10, 1, 12], [-8, 1, 14], [-6, 1, 16]])
        self.launcher.launch(self.settings.with_updates(wave_width=4))
        narrow = self.state.velocities.to_numpy()[:7].copy()
        self.launcher.launch(self.settings.with_updates(wave_width=12))
        added = self.state.velocities.to_numpy()[:7] - narrow
        self.assertGreater(np.count_nonzero(added[:, 0]), np.count_nonzero(narrow[:, 0]))
        self.assertEqual(self.state.count[None], 7)

    def test_forward_speed_and_lift_are_independent_live_controls(self):
        self.upload([[-12, 2, 12]])
        slow = self.settings.with_updates(wave_speed=2, wave_height=0)
        self.launcher.launch(slow)
        np.testing.assert_array_equal(self.state.velocities[0], [2, 0, 0])
        fast = slow.with_updates(wave_speed=8)
        self.launcher.launch(fast)
        np.testing.assert_array_equal(self.state.velocities[0], [10, 0, 0])
        lift_only = fast.with_updates(wave_speed=0, wave_height=2)
        self.launcher.launch(lift_only)
        np.testing.assert_allclose(self.state.velocities[0], [10, sqrt(4 * fast.gravity), 0], rtol=2e-6)

    def test_height_lift_energy_and_zero_gravity_reference(self):
        for gravity in (0, 9.81):
            lifts = []
            for height in (0, 0.5, 2):
                self.upload([[-12, 2, 12]])
                settings = self.settings.with_updates(gravity=gravity, wave_speed=0, wave_height=height)
                self.launcher.launch(settings)
                velocity = np.asarray(self.state.velocities[0])
                np.testing.assert_allclose(velocity, [0, sqrt(2 * max(gravity, REFERENCE_ACCELERATION) * height), 0],
                                           atol=2e-6)
                lifts.append(velocity[1])
            self.assertGreater(lifts[1], 0)
            self.assertAlmostEqual(lifts[2], 2 * lifts[1], places=5)

    def test_repeated_launches_bound_combined_three_dimensional_velocity(self):
        self.upload([[-12, 2, 12], [-14, 2, 12], [-8, 2, 12]],
                    [[-20, -10, 45], [30, -15, -35], [2, 3, 4]])
        settings = self.settings.with_updates(wave_speed=MAX_SPEED, wave_height=10)
        initial = self.state.velocities.to_numpy()[:3].copy()
        for launch in range(50):
            self.launcher.launch(settings)
            actual = self.state.velocities.to_numpy()[:3]
            if launch == 0:
                expected = initial[0] + [MAX_SPEED, sqrt(2 * settings.gravity * 10), 0]
                expected *= MAX_SPEED / np.linalg.norm(expected)
                np.testing.assert_allclose(actual[0], expected, atol=3e-6)
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.all(np.linalg.norm(actual, axis=1) <= MAX_SPEED + 1e-5))
            np.testing.assert_array_equal(actual[2], initial[2])
        self.assertGreater(actual[0, 0], 0)
        self.assertEqual(self.state.count[None], 3)

    def test_zero_strength_is_a_no_op_even_with_existing_motion(self):
        self.upload([[-12, 2, 12]], [[30, 20, -10]])
        before = self.state.velocities.to_numpy()
        self.launcher.launch(self.settings.with_updates(wave_speed=0, wave_height=0))
        np.testing.assert_array_equal(self.state.velocities.to_numpy(), before)

    def test_launch_after_reset_uses_new_count_without_downloads_or_allocations(self):
        count = self.state.active_count
        # An empty prefix and a restored lattice exercise the same compiled kernel.
        self.state.count[None] = 0
        self.launcher.launch()
        np.testing.assert_array_equal(self.state.velocities.to_numpy(), 0)
        self.state.reset(self.settings.with_updates(water_depth=1))
        self.assertEqual(self.state.active_count, 2 * count)
        with (patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("download")),
              patch.object(ti.Vector, "field", side_effect=AssertionError("allocation")),
              patch.object(ti, "field", side_effect=AssertionError("allocation"))):
            self.launcher.launch()
        velocity = self.state.velocities.to_numpy()
        self.assertTrue((velocity[:2 * count, 0] > 0).any())
        np.testing.assert_array_equal(velocity[2 * count:], 0)
        self.assertEqual(self.state.count[None], 2 * count)


if __name__ == "__main__":
    unittest.main()
