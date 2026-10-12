"""Check candidate accounting and warmed native timing without a desktop."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fluids import app, benchmark
from fluids.config import Settings


class FakeScalar:
    def __getitem__(self, key):
        return 0


class BenchmarkTests(unittest.TestCase):
    def test_scalar_counter_observes_only_target_fields_and_restores_getter(self):
        sim = SimpleNamespace(grid=SimpleNamespace(overflow_count=FakeScalar()),
                              dynamics=SimpleNamespace(_step_limit=FakeScalar()))
        original = FakeScalar.__getitem__
        with benchmark._count_scalar_reads(sim) as reads:
            sim.grid.overflow_count[None]
            sim.dynamics._step_limit[None]
            sim.dynamics._step_limit[None]
            FakeScalar()[None]
        self.assertEqual(reads, {"overflow": 1, "timestep": 2})
        self.assertIs(FakeScalar.__getitem__, original)

    def test_candidate_counts_match_explicit_cell_pairs_including_edges(self):
        counts = np.arange(24).reshape(4, 3, 2) % 5
        for span in (1, 2):
            expected = sum(int(counts[a]) * int(counts[b])
                           for a in np.ndindex(counts.shape) for b in np.ndindex(counts.shape)
                           if max(abs(x - y) for x, y in zip(a, b)) <= span)
            self.assertEqual(benchmark.candidate_checks(counts, span), expected)

    def test_native_excludes_first_frame_and_reports_simulated_and_discarded_time(self):
        settings = Settings()
        sim = SimpleNamespace(applied_settings=settings, paused=False, last_error=None,
                              simulated_time=0.0, discarded_time=0.0, accumulated_time=0.001)
        sim.grid = SimpleNamespace(overflow_count=FakeScalar())
        sim.dynamics = SimpleNamespace(_step_limit=FakeScalar())

        def advance(*args):
            sim.grid.overflow_count[None]
            sim.dynamics._step_limit[None]
            sim.simulated_time += 0.01
            sim.discarded_time += 0.02
            return 2

        sim.advance = advance

        class FakeApplication:
            def __init__(self, window, simulation):
                self.window = window
                self.controls = SimpleNamespace(settings=simulation.applied_settings)

            def warm_up(self):
                pass

            def run(self):
                self.warm_up()
                while self.window.running:
                    app.monotonic()
                    if not sim.paused:
                        sim.advance(0.02, settings)
                    app.monotonic()

        # First frame takes 100s, followed by 8s warm-up and two 0.5s frames.
        # Each advance takes 0.1s; these are nested inside the frame clocks.
        times = [0, 0.1, 0.2, 100, 108, 108.1, 108.2, 108.5,
                 108.5, 108.6, 108.7, 109]
        window = Mock(running=True)
        with (patch.object(app, "create_window", return_value=window),
              patch.object(app, "Application", FakeApplication),
              patch.object(benchmark, "perf_counter", side_effect=times),
              patch.object(benchmark.ti, "sync", side_effect=AssertionError("extra sync"))):
            result = benchmark.measure_native(sim, warmup_seconds=8, sample_seconds=1)
        self.assertTrue(result["sample_complete"])
        self.assertEqual(result["frame"]["samples"], 2)
        self.assertEqual(result["fps"], 2)
        self.assertEqual(result["sample_seconds"], 1)
        self.assertAlmostEqual(result["physics_advance_wall"]["mean_ms"], 100)
        self.assertAlmostEqual(result["simulated_seconds"], 0.02)
        self.assertAlmostEqual(result["discarded_seconds"], 0.04)
        self.assertEqual(result["substeps"], 4)
        self.assertEqual(result["scalar_reads"]["counts"], {"overflow": 2, "timestep": 2})
        self.assertEqual(result["scalar_reads"]["total"], 4)
        window.destroy.assert_called_once()

    def test_native_cpu_is_rejected_before_backend_initialization(self):
        with (patch.object(benchmark, "initialize_gpu") as initialize,
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            benchmark.main(["--mode", "native", "--backend", "cpu"])
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
