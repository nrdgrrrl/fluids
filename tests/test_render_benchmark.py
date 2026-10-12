"""Benchmark sampling/control contracts; these never measure native GPU FPS."""

import contextlib
import io
import unittest
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import taichi as ti
from taichi._lib import core as ti_core
from taichi.ui.scene import SceneV2

from fluids import render_benchmark as benchmark
from fluids.config import Settings
from fluids.controls import CameraController
from fluids.rendering import OUTLINE_COLOR, WATER_COLOR, Renderer
from fluids.simulation import Simulation
from fluids.world import generate_world


class FrameSampleTests(unittest.TestCase):
    def test_compilation_first_frame_and_warmup_are_excluded(self):
        samples = benchmark.FrameSamples(warmup=8, seconds=1)
        sim = SimpleNamespace(last_substeps=2)
        samples.begin(0)
        self.assertFalse(samples.finish(30, sim))  # first frame compilation
        samples.begin(37.9)
        self.assertFalse(samples.finish(38.1, sim))  # straddles warm-up end
        samples.begin(38.1)
        samples.physics, samples.submission = 0.2, 0.05
        self.assertFalse(samples.finish(38.5, sim))
        samples.begin(38.6)
        samples.physics, samples.submission = 0.3, 0.1
        self.assertTrue(samples.finish(39.2, sim))
        report = samples.report()
        self.assertEqual(report["frames"], 2)
        self.assertAlmostEqual(report["sample_seconds"], 1.1)
        self.assertAlmostEqual(report["fps"], 2 / 1.1)
        self.assertAlmostEqual(report["mean_frame_ms"], 550)
        self.assertAlmostEqual(report["mean_physics_advance_ms"], 250)
        self.assertAlmostEqual(report["mean_render_submission_ms"], 75)
        self.assertEqual(report["mean_substeps"], 2)

    def test_no_samples_is_an_explicit_failure(self):
        with self.assertRaisesRegex(RuntimeError, "before any warmed samples"):
            benchmark.FrameSamples(8, 30).report()

    def test_cli_backend_failure_does_not_create_window_or_write_report(self):
        with (
            patch.object(benchmark, "initialize_gpu", side_effect=RuntimeError("unsupported Vulkan")),
            patch.object(benchmark, "create_window") as window,
            patch.object(benchmark, "run_benchmark") as run,
            patch.object(ti, "reset"),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(benchmark.main([]), 1)
        self.assertIn("unsupported Vulkan", stderr.getvalue())
        window.assert_not_called()
        run.assert_not_called()

    def test_invalid_options_fail_before_initialization(self):
        for args in (["--seconds", "0"], ["--warmup", "nan"], ["--seconds", "inf"],
                     ["--resets", "-1"], ["--refresh-hz", "-1"]):
            with (self.subTest(args=args), patch.object(benchmark, "initialize_gpu") as initialize,
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                benchmark.main(args)
            initialize.assert_not_called()


class BenchmarkBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_cpu_benchmark_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "requires Vulkan"):
            benchmark.run_benchmark(Mock())

    def test_stock_control_uses_original_public_calls_without_device_vbos(self):
        world = generate_world(Settings())
        with patch.object(ti.Vector, "ndarray", side_effect=AssertionError("stock device VBO")):
            renderer = benchmark.StockRenderer(world)
        positions = ti.Vector.field(3, ti.f32, shape=32)
        colors = ti.Vector.field(3, ti.f32, shape=32)
        scene = Mock()
        renderer.render(scene, particle_positions=positions, particle_colors=colors,
                        particle_count=5, particle_radius=0.2)
        scene.mesh.assert_called_once_with(renderer.vertices, normals=renderer.normals,
                                          per_vertex_color=renderer.colors, two_sided=True,
                                          vertex_count=renderer.vertex_count)
        scene.lines.assert_called_once_with(renderer.outline_vertices, width=2.0, color=OUTLINE_COLOR)
        scene.particles.assert_called_once_with(positions, radius=0.2, color=WATER_COLOR,
                                               per_vertex_color=colors, index_count=5)
        self.assertEqual(renderer.staging_info()["particle_host_vbo_bytes_per_draw"], 32 * 48)
        self.assertFalse(renderer.update_world(generate_world(Settings())))

    def test_device_diagnostic_rejects_host_source_and_excess_transfer(self):
        renderer = Renderer(generate_world(Settings()))
        positions = ti.Vector.field(3, ti.f32, shape=32)
        renderer.render(SceneV2(Mock(spec=ti_core.PySceneV2)), particle_positions=positions, particle_count=5)
        benchmark._verify_device_submission(renderer, 5)
        renderer._particle_info.field_source = ti_core.FieldSource.HostMappedPtr
        with self.assertRaisesRegex(RuntimeError, "not a device ndarray"):
            benchmark._verify_device_submission(renderer, 5)
        renderer._particle_info.field_source = ti_core.FieldSource.TaichiNDarray
        renderer._particle_info.num_elements = 32 * 12
        with self.assertRaisesRegex(RuntimeError, "beyond the active prefix"):
            benchmark._verify_device_submission(renderer, 5)

    def test_reset_runner_preserves_buffers_camera_and_original_pause(self):
        settings = Settings(quality="low", water_depth=0.5)
        sim = Simulation(settings)
        renderer = Renderer(sim.world)
        scene = SceneV2(Mock(spec=ti_core.PySceneV2))
        renderer.render(scene, particle_positions=sim.particles.positions,
                        particle_colors=sim.particles.colors, particle_count=sim.particles.active_count)
        app = SimpleNamespace(renderer=renderer, simulation=sim, scene=scene, camera=Mock(),
                              controls=SimpleNamespace(camera=CameraController()),
                              window=Mock(running=True), canvas=Mock())
        original_camera = benchmark._camera_state(app)
        sim.paused = False
        # Reset-runner clocks/RSS use real CPU operations, but their numeric
        # values are deliberately not asserted or reported as GPU measurements.
        results = benchmark._reset_checks(app, settings, 10)
        self.assertEqual(len(results), 10)
        self.assertEqual([row["reset"] for row in results], list(range(1, 11)))
        self.assertEqual(len({row["particles"] for row in results}), 2)
        self.assertFalse(sim.paused)
        self.assertEqual(benchmark._camera_state(app), original_camera)
        self.assertEqual(renderer._particle_allocations, 1)
        self.assertEqual(app.window.show.call_count, 10)
        app.window.show.assert_called_with()

    def test_sampling_wraps_real_application_loop_with_mocked_desktop_and_clock(self):
        # Only the benchmark's backend gate is mocked. Real simulation/packing
        # kernels still run on CPU and the native scene never renders. Synthetic
        # times below test sampling/termination, not hardware performance.
        backend_gate = SimpleNamespace(cfg=SimpleNamespace(arch=ti.vulkan), vulkan=ti.vulkan,
                                       Vector=ti.Vector, field=ti.field, f32=ti.f32, sync=ti.sync)
        for staging in ("device", "stock"):
            window = Mock(running=True)
            window.get_window_shape.return_value = (1440, 900)
            window.get_cursor_pos.return_value = (0.7, 0.5)
            window.is_pressed.return_value = False
            window.get_events.return_value = []
            window.get_scene.return_value = SceneV2(Mock(spec=ti_core.PySceneV2))
            gui = window.get_gui.return_value
            gui.sub_window.side_effect = lambda *args: nullcontext(gui)
            gui.button.return_value = False
            gui.checkbox.side_effect = lambda label, selected: selected
            gui.slider_int.side_effect = lambda label, value, lo, hi: value
            gui.slider_float.side_effect = lambda label, value, lo, hi: value
            clock = count()
            with (
                self.subTest(staging=staging),
                patch.object(benchmark, "ti", backend_gate),
                patch.object(benchmark, "DEFAULT_SETTINGS", Settings(quality="low", water_depth=0.5)),
                patch.object(benchmark, "perf_counter", side_effect=lambda: next(clock) / 10),
                patch.object(benchmark, "_rss_bytes", return_value=None),
            ):
                report = benchmark.run_benchmark(window, staging=staging, quality="low", scenario="paused",
                                                 warmup=0.1, seconds=1.0, resets=0,
                                                 capture=Path("synthetic-test.png"))
            self.assertGreater(report["timing"]["frames"], 0)
            self.assertGreaterEqual(report["timing"]["sample_seconds"], 1)
            self.assertEqual(report["timing"]["mean_physics_advance_ms"], 0)
            self.assertEqual(report["simulation_end"]["simulated_seconds"], 0)
            self.assertEqual(report["water_city_resets"], [])
            self.assertIn("unmeasured", report["gpu_memory"])
            window.save_image.assert_called_once_with("synthetic-test.png")


if __name__ == "__main__":
    unittest.main()
