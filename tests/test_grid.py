"""Real binning kernels and neighbor searches; downloads are test inspection only.

CPU is the default for headless CI. FLUIDS_TEST_PARTICLE_ARCH=vulkan runs these
checks on Vulkan with fallback disabled, like the particle-field tests.
"""

import os
import unittest
from dataclasses import replace
from itertools import product
from unittest.mock import Mock, patch

import numpy as np
import taichi as ti

from fluids.config import PARTICLE_SPACING, SMOOTHING_LENGTH_FACTOR, Settings
from fluids.grid import CELL_CAPACITY, CELL_SIZE, GRID_SHAPE, SpatialGrid
from fluids.particles import ParticleState
from fluids.world import generate_world


@ti.kernel
def map_cells(grid: ti.template(), positions: ti.template(), output: ti.template(), count: ti.i32):
    for i in range(count):
        output[i] = grid.cell_coordinates(positions[i])


@ti.data_oriented
class NeighborProbe:
    def __init__(self, capacity):
        self.visits = ti.field(dtype=ti.i32, shape=(capacity, capacity))

    @ti.func
    def record(self, i, j):
        self.visits[i, j] += 1

    @ti.kernel
    def run(self, grid: ti.template(), count: ti.i32):
        for i in range(count):
            grid.for_each_neighbor(i, self.record)


@ti.data_oriented
class SampledNeighborProbe:
    """Compare full neighbor sets of selected queries without an N-by-N field."""

    def __init__(self, capacity, queries):
        self.queries = ti.field(dtype=ti.i32, shape=len(queries))
        self.queries.from_numpy(np.asarray(queries, dtype=np.int32))
        self.rows = ti.field(dtype=ti.i32, shape=capacity)
        self.rows.fill(-1)
        for row, i in enumerate(queries):
            self.rows[int(i)] = row
        self.visits = ti.field(dtype=ti.i32, shape=(len(queries), capacity))

    @ti.func
    def record(self, i, j):
        self.visits[self.rows[i], j] += 1

    @ti.kernel
    def run(self, grid: ti.template()):
        for row in range(self.queries.shape[0]):
            grid.for_each_neighbor(self.queries[row], self.record)


class SpatialGridTests(unittest.TestCase):
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

    def upload(self, points):
        points = np.asarray(points, dtype=np.float32)
        positions = np.zeros((self.state.capacity, 3), dtype=np.float32)
        positions[:len(points)] = points
        self.state.positions.from_numpy(positions)
        self.state.count[None] = len(points)
        return points

    def stored_indices(self):
        counts = self.grid.cell_counts.to_numpy().reshape(-1)
        indices = self.grid.particle_indices.to_numpy().reshape(-1, CELL_CAPACITY)
        return indices[np.arange(CELL_CAPACITY)[None, :] < counts[:, None]]

    def test_grid_dimensions_and_all_supported_radii(self):
        self.assertEqual(self.grid.cell_counts.shape, (48, 16, 32))
        self.assertEqual(self.grid.particle_indices.shape, (48, 16, 32, 256))
        self.assertEqual(CELL_SIZE, 1.0)
        self.assertEqual(self.grid.storage_bytes, 25_264_156)
        self.assertLessEqual(max(PARTICLE_SPACING.values()) * SMOOTHING_LENGTH_FACTOR, CELL_SIZE)

    def test_supported_extremes_insert_once_without_overflow(self):
        fields = (self.grid.cell_counts, self.grid.particle_indices, self.grid.overflow_count)
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 4.0):
                with self.subTest(quality=quality, depth=depth):
                    self.state.reset(self.settings.with_updates(quality=quality, water_depth=depth))
                    positions = self.state.lattice.positions()
                    coords = np.floor((positions - self.state.world.domain_min) / CELL_SIZE).astype(int)
                    expected = np.zeros(GRID_SHAPE, dtype=np.int32)
                    np.add.at(expected, tuple(coords.T), 1)
                    expected_overflow = int(np.maximum(expected - CELL_CAPACITY, 0).sum())
                    self.assertEqual(expected_overflow, 0)
                    self.assertEqual(self.grid.build(), expected_overflow)
                    counts = self.grid.cell_counts.to_numpy()
                    self.assertEqual(int(counts.sum()), self.state.active_count)
                    np.testing.assert_array_equal(counts, expected)
                    stored = self.stored_indices()
                    self.assertEqual(len(stored), self.state.active_count - expected_overflow)
                    self.assertEqual(len(np.unique(stored)), len(stored))
                    self.assertTrue(((stored >= 0) & (stored < self.state.active_count)).all())
                    np.testing.assert_array_equal(np.sort(stored), np.arange(self.state.active_count))
        self.assertIs(self.grid.cell_counts, fields[0])
        self.assertIs(self.grid.particle_indices, fields[1])
        self.assertIs(self.grid.overflow_count, fields[2])

    def test_boundary_positions_map_to_valid_cells(self):
        origin = np.asarray(self.state.world.domain_min, dtype=np.float32)
        upper = np.asarray(self.state.world.domain_max, dtype=np.float32)
        grid_upper = origin + np.asarray(GRID_SHAPE, dtype=np.float32) * CELL_SIZE
        corners = list(product(*zip(origin, upper)))
        grid_corners = list(product(*zip(origin, grid_upper)))
        points = self.upload(corners + grid_corners + [origin - 1e-5, grid_upper + 1e-5])
        self.assertEqual(self.grid.build(), 0)
        cells = ti.Vector.field(3, dtype=ti.i32, shape=len(points))
        map_cells(self.grid, self.state.positions, cells, len(points))
        actual = cells.to_numpy()
        expected = np.floor((points - origin) / CELL_SIZE).astype(int)
        expected = np.clip(expected, 0, np.asarray(GRID_SHAPE) - 1)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue((actual >= 0).all())
        self.assertTrue((actual < GRID_SHAPE).all())
        np.testing.assert_array_equal(np.sort(self.stored_indices()), np.arange(len(points)))

    def test_overflow_is_exact_and_never_writes_outside_a_cell(self):
        # Sentinel-filled adjacent/unused cells expose writes past a cell's slots.
        self.grid.particle_indices.fill(-1)
        origin = np.asarray(self.state.world.domain_min)
        for count in (CELL_CAPACITY, CELL_CAPACITY + 1, 4 * CELL_CAPACITY + 17):
            with self.subTest(count=count):
                self.upload(np.tile(origin + CELL_SIZE * 0.5, (count, 1)))
                self.assertEqual(self.grid.build(), max(0, count - CELL_CAPACITY))
                self.assertEqual(self.grid.overflow_count[None], max(0, count - CELL_CAPACITY))
                counts = self.grid.cell_counts.to_numpy()
                self.assertEqual(counts[0, 0, 0], count)
                self.assertEqual(int(counts.sum()), count)
                indices = self.grid.particle_indices.to_numpy()
                filled = indices[0, 0, 0]
                self.assertEqual(len(np.unique(filled)), CELL_CAPACITY)
                self.assertTrue(((filled >= 0) & (filled < count)).all())
                indices[0, 0, 0] = -1
                np.testing.assert_array_equal(indices, -1)

    def test_overflow_sums_omitted_insertions_across_cells(self):
        origin = np.asarray(self.state.world.domain_min)
        points = np.concatenate([
            np.tile(origin + [0.2, 0.2, 0.2], (CELL_CAPACITY + 3, 1)),
            np.tile(origin + [2.0, 0.2, 0.2], (CELL_CAPACITY + 5, 1)),
        ])
        self.upload(points)
        self.assertEqual(self.grid.build(), 8)

    def test_substep_rebuilds_skips_overflow_and_recovers_without_downloads(self):
        physics = Mock()
        origin = np.asarray(self.state.world.domain_min)
        with patch.object(type(self.state.positions), "to_numpy", side_effect=AssertionError("download")):
            self.assertEqual(self.grid.run_substep(physics, 0.01, gravity=9.81), 0)
            physics.assert_called_once_with(0.01, gravity=9.81)
            physics.reset_mock()
            self.upload(np.tile(origin + 0.2, (CELL_CAPACITY + 9, 1)))
            self.assertEqual(self.grid.run_substep(physics), 9)
            physics.assert_not_called()
            self.state.reset()
            self.assertEqual(self.grid.run_substep(physics), 0)
            physics.assert_called_once_with()
        self.assertEqual(int(self.grid.cell_counts.to_numpy().sum()), self.state.active_count)

    def test_rebuild_clears_counts_and_ignores_inactive_and_stale_indices(self):
        self.assertEqual(self.grid.build(), 0)
        origin = np.asarray(self.state.world.domain_min)
        # Inactive positions deliberately remain in a previously occupied cell.
        self.state.positions.fill(tuple(origin + 0.2))
        self.state.count[None] = 1
        self.assertEqual(self.grid.build(), 0)
        np.testing.assert_array_equal(self.stored_indices(), [0])
        self.state.count[None] = 0
        self.assertEqual(self.grid.build(), 0)
        np.testing.assert_array_equal(self.grid.cell_counts.to_numpy(), 0)

    def test_neighbor_search_matches_brute_force_with_no_duplicate_visits(self):
        origin = np.asarray(self.state.world.domain_min)
        rng = np.random.default_rng(6)
        # Include diagonal neighbors at an interior cell corner and at grid edges.
        points = np.concatenate([
            origin + 4 * CELL_SIZE + np.asarray(list(product((-0.2, 0.2), repeat=3))),
            origin + rng.uniform(0, 3 * CELL_SIZE, size=(60, 3)),
            origin + np.asarray(GRID_SHAPE) * CELL_SIZE - rng.uniform(0, 0.5, size=(8, 3)),
        ])
        points = self.upload(points)
        probe = NeighborProbe(len(points))
        self.assertEqual(self.grid.run_substep(probe.run, self.grid, len(points)), 0)
        distances = points[:, None, :] - points[None, :, :]
        expected = (np.sum(distances ** 2, axis=2) <= self.state.smoothing_radius ** 2).astype(int)
        np.testing.assert_array_equal(probe.visits.to_numpy(), expected)

    def test_neighbor_radius_and_origin_follow_reset_in_compiled_kernels(self):
        probe = NeighborProbe(2)
        for quality, width in (("low", 40.0), ("high", 48.0), ("medium", 32.0)):
            with self.subTest(quality=quality):
                settings = self.settings.with_updates(quality=quality)
                self.state.reset(settings, generate_world(settings, domain_width=width))
                origin = np.asarray(self.state.world.domain_min)
                points = self.upload([origin + 0.2, origin + [1.0, 0.2, 0.2]])
                probe.visits.fill(0)
                self.assertEqual(self.grid.run_substep(probe.run, self.grid, 2), 0)
                distance = np.linalg.norm(points[0] - points[1])
                expected = np.eye(2, dtype=int)
                if distance <= self.state.smoothing_radius:
                    expected.fill(1)
                np.testing.assert_array_equal(probe.visits.to_numpy(), expected)
                cells = ti.Vector.field(3, dtype=ti.i32, shape=2)
                map_cells(self.grid, self.state.positions, cells, 2)
                expected_cells = np.floor((points - origin) / self.grid.cell_size).astype(int)
                np.testing.assert_array_equal(cells.to_numpy(), expected_cells)

    def test_full_lattice_sampled_neighbors_match_brute_force_at_all_qualities_and_depths(self):
        rng = np.random.default_rng(36)
        # Reuse queries, probe, and grid fields across every reset. Extremal
        # queries exercise the water's lower/upper faces as well as its interior.
        queries = np.arange(40, dtype=np.int32)
        probe = SampledNeighborProbe(self.state.capacity, queries)
        grids = (self.grid, SpatialGrid(self.state, cell_size=0.7))
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 2.0, 4.0):
                with self.subTest(quality=quality, depth=depth):
                    self.state.reset(self.settings.with_updates(quality=quality, water_depth=depth))
                    points = self.state.lattice.positions()
                    chosen = np.concatenate(([0, len(points) - 1],
                                             rng.choice(np.arange(1, len(points) - 1), 38, replace=False)))
                    probe.queries.from_numpy(chosen.astype(np.int32))
                    rows = np.full(self.state.capacity, -1, dtype=np.int32)
                    rows[chosen] = np.arange(len(chosen))
                    probe.rows.from_numpy(rows)
                    squared = np.sum((points[chosen, None] - points[None]) ** 2, axis=2)
                    expected = squared <= np.float32(self.state.smoothing_radius)**2
                    for grid in grids:
                        probe.visits.fill(0)
                        self.assertEqual(grid.run_substep(probe.run, grid), 0)
                        actual = probe.visits.to_numpy()
                        np.testing.assert_array_equal(actual[:, :len(points)], expected)
                        np.testing.assert_array_equal(actual[:, len(points):], 0)

    def test_quality_sized_layout_expands_stencil_and_reuses_fields_after_reset(self):
        grid = SpatialGrid(self.state, cell_size=0.5)
        fields = (grid.cell_counts, grid.particle_indices, grid._span)
        probe = NeighborProbe(4)
        for quality in ("high", "medium", "low", "high"):
            self.state.reset(self.settings.with_updates(quality=quality))
            # Low quality includes a neighbor two cell indices away. The exact
            # support boundary is included once and just-outside is excluded.
            points = self.upload([[-19.51, 0.1, 0.1], [-18.6, 0.1, 0.1],
                                  [-19.01, 0.1, 0.1], [-19.009, 0.1, 0.1]])
            probe.visits.fill(0)
            self.assertEqual(grid.run_substep(probe.run, grid, 4), 0)
            squared = np.sum((points[:, None] - points[None])**2, axis=2)
            np.testing.assert_array_equal(probe.visits.to_numpy(),
                                          squared <= np.float32(self.state.smoothing_radius)**2)
            with patch.object(ti, "field", side_effect=AssertionError("new field")):
                grid.reset()
        self.assertEqual(fields, (grid.cell_counts, grid.particle_indices, grid._span))

    def test_boundary_neighbor_sets_at_every_quality_and_depth(self):
        probe = NeighborProbe(24)
        for quality in PARTICLE_SPACING:
            for depth in (0.5, 2.0, 4.0):
                self.state.reset(self.settings.with_updates(quality=quality, water_depth=depth))
                origin = np.asarray(self.state.world.domain_min, dtype=np.float32)
                upper = np.asarray(self.state.world.domain_max, dtype=np.float32)
                radius = self.state.smoothing_radius
                corners = np.asarray(list(product(*zip(origin, upper))), dtype=np.float32)
                toward_inside = np.where(corners == origin, 1, -1)
                points = self.upload(np.concatenate((corners, corners + toward_inside * radius * 0.25,
                                                    corners + toward_inside * radius * 0.75)))
                probe.visits.fill(0)
                self.assertEqual(self.grid.run_substep(probe.run, self.grid, len(points)), 0)
                squared = np.sum((points[:, None] - points[None])**2, axis=2)
                np.testing.assert_array_equal(probe.visits.to_numpy(),
                                              squared <= np.float32(radius)**2)

    def test_incomplete_or_unbuilt_grid_cannot_supply_partial_neighbors(self):
        probe = NeighborProbe(1)
        probe.run(self.grid, 1)
        np.testing.assert_array_equal(probe.visits.to_numpy(), 0)
        self.upload(np.tile(np.asarray(self.state.world.domain_min) + 0.2, (CELL_CAPACITY + 1, 1)))
        self.assertEqual(self.grid.build(), 1)
        probe.run(self.grid, 1)
        np.testing.assert_array_equal(probe.visits.to_numpy(), 0)

    def test_unsupported_radius_or_domain_rejected_before_gpu_allocation(self):
        invalid_world = replace(self.state.world, domain_max=(40.0, 12.0, 24.0))
        with patch.object(ti, "field") as allocate:
            with patch.object(self.state, "world", invalid_world):
                with self.assertRaisesRegex(ValueError, "bounds"):
                    SpatialGrid(self.state)
            with patch.object(self.state, "lattice", replace(self.state.lattice, spacing=1.0)):
                with self.assertRaisesRegex(ValueError, "smoothing radius"):
                    SpatialGrid(self.state)
            for width in (0, 0.49, float("inf"), float("nan")):
                with self.assertRaisesRegex(ValueError, "cell size"):
                    SpatialGrid(self.state, cell_size=width)
        allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
