"""Contact acceptance tests using real Taichi kernels, with optional Vulkan.

Set FLUIDS_TEST_PARTICLE_ARCH=vulkan to exercise the production backend.
Particle downloads in this module are test inspection only.
"""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import taichi as ti

from fluids.collisions import CollisionSolver, MAX_COLLISION_RADIUS, PLOT_SHAPE, _geometry_arrays
from fluids.config import Settings
from fluids.dynamics import DynamicsSolver
from fluids.grid import SpatialGrid
from fluids.particles import ParticleState
from fluids.world import generate_world


class CollisionGeometryTests(unittest.TestCase):
    def test_geometry_guards_preserve_single_plot_lookup(self):
        world = generate_world(Settings(city_density=1))
        building = world.buildings[0]
        for invalid in (
            replace(building, minimum=(0.49, *building.minimum[1:])),
            replace(building, plot=(-1, 0)),
            replace(building, plot=None),
            replace(building, maximum=(building.maximum[0], 12.0, building.maximum[2])),
        ):
            with self.subTest(building=invalid), self.assertRaises(ValueError):
                _geometry_arrays(replace(world, buildings=(invalid,)))
        with self.assertRaisesRegex(ValueError, "one building"):
            _geometry_arrays(replace(world, buildings=(building, building)))

    def test_contact_controls_are_validated_and_live(self):
        settings = Settings()
        for name in ("collision_restitution", "collision_friction"):
            for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings.with_updates(**{name: value})
            for value in (0, 0.5, 1):
                self.assertEqual(settings.with_updates(**{name: value}).reset_signature(),
                                 settings.reset_signature())


class CollisionKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        arch = os.environ.get("FLUIDS_TEST_PARTICLE_ARCH", "cpu")
        if arch not in ("cpu", "vulkan"):
            raise ValueError("FLUIDS_TEST_PARTICLE_ARCH must be cpu or vulkan")
        ti.init(arch=getattr(ti, arch), enable_fallback=False, offline_cache=False, debug=True)
        cls.settings = Settings(quality="low", water_depth=0.5, city_density=1,
                                collision_restitution=0.5, collision_friction=0.25)
        cls.world = generate_world(cls.settings)
        cls.state = ParticleState(cls.settings, cls.world, capacity=20_000)
        cls.solver = CollisionSolver(cls.state)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def setUp(self):
        self.state.reset(self.settings, self.world)

    def upload(self, points, velocities=None):
        positions = np.zeros((self.state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        speeds = np.zeros_like(positions)
        if velocities is not None:
            speeds[:len(points)] = velocities
        self.state.positions.from_numpy(positions)
        self.state.velocities.from_numpy(speeds)
        self.state.count[None] = len(points)

    def assert_clear(self, points):
        radius = self.state.collision_radius
        self.assertTrue(np.all(points >= np.asarray(self.state.world.domain_min) + radius - 2e-6))
        self.assertTrue(np.all(points <= np.asarray(self.state.world.domain_max) - radius + 2e-6))
        for box in self.state.world.boxes:
            inside = np.all((points > np.asarray(box.minimum) - radius + 2e-6)
                            & (points < np.asarray(box.maximum) + radius - 2e-6), axis=1)
            self.assertFalse(inside.any(), f"penetration in {box}")

    def test_all_six_container_faces_and_outgoing_velocities(self):
        radius = self.state.collision_radius
        lower, upper = np.array(self.world.domain_min) + radius, np.array(self.world.domain_max) - radius
        points, velocities, expected = [], [], []
        for axis in range(3):
            for side, bounds in enumerate((lower, upper)):
                inward = 1 if side == 0 else -1
                for incoming in (True, False):
                    position = np.array([-10.0, 5.0, 10.0])
                    position[axis] = bounds[axis] - inward * 0.1
                    velocity = np.array([1.0, 2.0, 3.0])
                    velocity[axis] = (-2 if incoming else 2) * inward
                    after = velocity.copy()
                    if incoming:
                        after *= 0.75
                        after[axis] = inward
                    points.append(position)
                    velocities.append(velocity)
                    position = position.copy()
                    position[axis] = bounds[axis]
                    expected.append((position, after))
        self.upload(points, velocities)
        self.solver.resolve()
        np.testing.assert_allclose(self.state.positions.to_numpy()[:len(points)], [x[0] for x in expected], atol=2e-6)
        np.testing.assert_allclose(self.state.velocities.to_numpy()[:len(points)], [x[1] for x in expected], atol=1e-6)
        self.assert_clear(self.state.positions.to_numpy()[:len(points)])

    def test_building_projects_to_nearest_accessible_expanded_face(self):
        box = self.world.buildings[0]
        radius = self.state.collision_radius
        lower, upper = np.array(box.minimum) - radius, np.array(box.maximum) + radius
        points, velocities, expected = [], [], []
        for axis, side in ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)):
            for incoming in (True, False):
                point = (lower + upper) / 2
                # Stay clear of land even for the smallest building heights.
                point[1] = max(point[1], self.world.land.maximum[1] + radius + 0.01)
                face = (lower, upper)[side][axis]
                direction = -1 if side == 0 else 1
                point[axis] = face - direction * 0.01
                velocity = np.zeros(3)
                velocity[axis] = (-2 if incoming else 2) * direction
                points.append(point.copy())
                velocities.append(velocity)
                point[axis] = face
                velocity = velocity.copy()
                if incoming:
                    velocity[axis] = direction
                expected.append((point, velocity))
        self.upload(points, velocities)
        self.solver.resolve()
        np.testing.assert_allclose(self.state.positions.to_numpy()[:len(points)], [x[0] for x in expected], atol=2e-6)
        np.testing.assert_allclose(self.state.velocities.to_numpy()[:len(points)], [x[1] for x in expected], atol=1e-6)
        self.assert_clear(self.state.positions.to_numpy()[:len(points)])

    def test_land_coast_and_top_with_buried_or_container_faces(self):
        self.state.reset(self.settings, generate_world(self.settings.with_updates(city_density=0)))
        points = [[-0.2, 0.3, 5], [10, 0.3, 5], [19.9, 0.1, 0.1], [10, -10, 10]]
        self.upload(points, [[2, 0, 0], [0, -2, 0], [0, -2, 0], [0, -2, 0]])
        self.solver.resolve()
        after = self.state.positions.to_numpy()[:len(points)]
        np.testing.assert_allclose(after[0], [-0.25, 0.3, 5], atol=1e-6)
        np.testing.assert_allclose(after[1:], [[10, 0.5, 5], [19.75, 0.5, 0.25], [10, 0.5, 10]], atol=2e-6)
        np.testing.assert_allclose(self.state.velocities.to_numpy()[:len(points)],
                                   [[-1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assert_clear(after)

    def test_overlapping_land_and_building_do_not_trap_particles(self):
        world = generate_world(self.settings.with_updates(building_min_height=0.5, building_max_height=0.5))
        self.state.reset(self.settings, world)
        points = [(np.array(box.minimum) + box.maximum) / 2 for box in world.buildings]
        points += [[point[0], 0.1, point[2]] for point in points]
        self.upload(points)
        self.solver.resolve()
        self.assert_clear(self.state.positions.to_numpy()[:len(points)])

    def test_random_deep_penetration_and_corners_at_every_quality(self):
        rng = np.random.default_rng(9)
        points = rng.uniform([-21, -1, -1], [21, 13, 25], (1000, 3))
        velocities = rng.uniform(-10, 10, points.shape)
        for quality in ("low", "medium", "high"):
            with self.subTest(quality=quality):
                self.state.reset(self.settings.with_updates(quality=quality))
                self.upload(points, velocities)
                self.solver.resolve()
                after = self.state.positions.to_numpy()[:len(points)]
                self.assert_clear(after)
                self.assertTrue(np.isfinite(self.state.velocities.to_numpy()[:len(points)]).all())
                self.solver.resolve()
                np.testing.assert_array_equal(self.state.positions.to_numpy()[:len(points)], after)

    def test_live_restitution_and_friction_extremes_and_exact_contact(self):
        point = [-19.75, 5, 10]
        for bounce in (0, 0.5, 1):
            for friction in (0, 0.5, 1):
                with self.subTest(bounce=bounce, friction=friction):
                    self.upload([point], [[-2, 3, 4]])
                    self.solver.resolve(self.settings.with_updates(collision_restitution=bounce,
                                                                   collision_friction=friction))
                    np.testing.assert_allclose(self.state.velocities[0], [2 * bounce, 3 * (1 - friction), 4 * (1 - friction)])

    def test_slots_reset_to_empty_without_reallocation_or_particle_download(self):
        self.solver.resolve()
        np.testing.assert_array_equal(self.solver.building_occupied.to_numpy(), np.ones(PLOT_SHAPE))
        self.assertEqual(self.solver.building_minimum.shape, PLOT_SHAPE)
        for box in self.world.buildings:
            np.testing.assert_allclose(self.solver.building_minimum[box.plot], box.minimum, atol=1e-6)
        vacant_world = generate_world(self.settings.with_updates(city_density=0))
        self.state.reset(self.settings, vacant_world)
        point = (np.asarray(self.world.buildings[0].minimum) + self.world.buildings[0].maximum) / 2
        self.upload([point], [[1, 2, 3]])
        self.state.positions[1], self.state.velocities[1] = [99, 99, 99], [4, 5, 6]
        with (
            patch.object(ti.Vector, "field", side_effect=AssertionError("reallocation")),
            patch.object(ti, "field", side_effect=AssertionError("reallocation")),
            patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("download")),
        ):
            self.solver.resolve()
        np.testing.assert_allclose(self.state.positions[0], point, atol=1e-6)
        np.testing.assert_array_equal(self.state.velocities[0], [1, 2, 3])
        np.testing.assert_array_equal(self.solver.building_occupied.to_numpy(), 0)
        np.testing.assert_array_equal(self.solver.building_minimum.to_numpy(), 0)
        np.testing.assert_array_equal(self.state.positions[1], [99, 99, 99])
        np.testing.assert_array_equal(self.state.velocities[1], [4, 5, 6])
        # Upload a sparse world after emptying slots, without assuming tuple order.
        sparse = replace(self.world, buildings=tuple(reversed(self.world.buildings[::4])))
        self.state.reset(self.settings, sparse)
        self.solver.resolve()
        occupied = self.solver.building_occupied.to_numpy()
        for box in self.world.buildings:
            self.assertEqual(occupied[box.plot], int(box in sparse.buildings))

    def test_unsafe_radius_is_rejected(self):
        self.state.lattice = replace(self.state.lattice, spacing=2 * (MAX_COLLISION_RADIUS + 0.01))
        with self.assertRaisesRegex(ValueError, "radius"):
            self.solver.resolve()

    def test_maximum_radius_and_changed_world_bounds_are_runtime_inputs(self):
        world = generate_world(self.settings, domain_width=32, land_height=0.3)
        self.state.reset(self.settings, world)
        self.state.lattice = replace(self.state.lattice, spacing=2 * MAX_COLLISION_RADIUS)
        points = [(np.array(box.minimum) + box.maximum) / 2 for box in world.buildings]
        points += [[-17, 15, -1], [17, 0, 25]]
        self.upload(points)
        self.solver.resolve()
        self.assert_clear(self.state.positions.to_numpy()[:len(points)])
        np.testing.assert_allclose(self.state.positions[len(points) - 2], [-15.64, 11.64, 0.36], atol=2e-6)

    def test_dynamics_resolves_contacts_after_each_integration(self):
        dynamics = DynamicsSolver(SpatialGrid(self.state))
        settings = self.settings.with_updates(gravity=0, viscosity=0, damping=0,
                                             collision_restitution=1, collision_friction=0)
        self.upload([[-19.749, 5, 10]], [[-1, 2, 0]])
        self.assertEqual(dynamics.step(0.005, settings), 0)
        np.testing.assert_allclose(self.state.positions[0], [-19.75, 5.01, 10], atol=1e-6)
        np.testing.assert_array_equal(self.state.velocities[0], [1, 2, 0])
        with (
            patch.object(dynamics.collisions, "resolve", wraps=dynamics.collisions.resolve) as resolve,
            patch.object(dynamics, "_integrate", wraps=dynamics._integrate) as integrate,
        ):
            self.assertEqual(dynamics.advance(0.03, settings), 0)
        self.assertGreater(resolve.call_count, 1)
        self.assertEqual(resolve.call_count, integrate.call_count)


if __name__ == "__main__":
    unittest.main()
