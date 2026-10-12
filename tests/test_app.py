"""Application integration, playback ordering, recovery, and frame timing."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import taichi as ti
from taichi._lib import core as ti_core

from fluids.app import Application
from fluids.config import Settings
from fluids.simulation import Simulation, SpatialGridOverflowError


def event(key, pressed=True):
    # Match the key-only native GGUI Event, with its kind supplied by a filter.
    return (ti.ui.PRESS if pressed else ti.ui.RELEASE, Mock(spec_set=ti_core.Event, key=key))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.window = Mock(running=True)
        self.window.get_window_shape.return_value = (1440, 900)
        self.window.get_cursor_pos.return_value = (0.7, 0.5)
        self.window.is_pressed.return_value = False
        self.gui = self.window.get_gui.return_value
        self.gui.sub_window.side_effect = lambda *args: nullcontext(self.gui)
        self.gui.button.return_value = False
        self.gui.checkbox.side_effect = lambda label, selected: selected
        self.gui.slider_int.side_effect = lambda label, value, lo, hi: value
        self.gui.slider_float.side_effect = lambda label, value, lo, hi: value
        self.settings = Settings(quality="low", water_depth=0.5)
        self.simulation = Mock(
            applied_settings=self.settings, paused=False, last_error=None,
            simulated_time=0.0, discarded_time=0.0, last_substeps=0,
            particles=SimpleNamespace(
                positions=object(), colors=object(), active_count=100,
                collision_radius=0.25,
            ),
        )
        self.simulation.requires_reset.side_effect = (
            lambda settings: settings.reset_signature()
            != self.simulation.applied_settings.reset_signature()
        )

        def reset(settings):
            self.simulation.applied_settings = settings
            self.simulation.last_error = None

        self.simulation.reset.side_effect = reset
        self.renderer = Mock()
        self.camera = Mock()
        self.enterContext(patch.object(ti.ui, "Camera", return_value=self.camera))
        self.app = Application(self.window, simulation=self.simulation, renderer=self.renderer)

    def set_event_batches(self, batches):
        self.window.get_events.side_effect = [
            [item for kind, item in batch if kind is tag]
            for batch in batches
            for tag in (ti.ui.RELEASE, ti.ui.PRESS)
        ]

    def run_frames(self, batches):
        self.set_event_batches([*batches, [event("Escape")]])
        with patch.object(self.app, "warm_up"):
            self.app.run()

    def test_first_panel_click_keeps_rendering_and_dispatches_button(self):
        self.window.get_cursor_pos.return_value = (0.1, 0.8)
        self.window.is_pressed.side_effect = lambda key: key == "LMB"
        self.gui.button.side_effect = lambda label: label == "Pause (Space)"
        self.run_frames([[event("LMB")]])
        self.assertTrue(self.simulation.paused)
        self.simulation.advance.assert_not_called()
        self.renderer.render.assert_called_once()
        self.window.show.assert_called_once()
        self.assertEqual(self.app.controls.camera.azimuth,
                         self.app.controls.camera.DEFAULT_AZIMUTH)

    def test_window_close_renders_complete_scene_and_keeps_paused_camera_responsive(self):
        self.simulation.paused = True
        self.window.get_events.return_value = []
        self.window.is_pressed.side_effect = lambda key: key == "RMB"
        self.window.get_cursor_pos.side_effect = [(0.7, 0.5), (0.8, 0.6)]
        def show():
            if self.window.show.call_count == 2:
                self.window.running = False

        self.window.show.side_effect = show
        with patch.object(self.app, "warm_up"):
            self.app.run()
        self.simulation.advance.assert_not_called()
        self.assertNotEqual(self.app.controls.camera.target,
                            self.app.controls.camera.DEFAULT_TARGET)
        self.assertEqual(self.camera.position.call_count, 2)
        self.renderer.render.assert_called_with(
            self.app.scene, particle_positions=self.simulation.particles.positions,
            particle_radius=0.25, particle_count=100,
            particle_colors=self.simulation.particles.colors,
        )
        self.app.canvas.scene.assert_called_with(self.app.scene)

    def test_panel_pause_precedes_physics_and_reset_precedes_render(self):
        calls = []
        self.gui.button.side_effect = lambda label: label in ("Pause (Space)", "Apply / reset (R)")
        self.app.controls.settings = self.settings.with_updates(city_density=0.2)

        def reset(settings):
            calls.append("reset")
            self.assertTrue(self.simulation.paused)
            self.simulation.applied_settings = settings

        self.simulation.reset.side_effect = reset
        self.renderer.update_world.side_effect = lambda world: calls.append("geometry")
        self.renderer.render.side_effect = lambda *args, **kwargs: calls.append("render")
        self.run_frames([[]])
        self.assertEqual(calls, ["reset", "geometry", "render"])
        self.simulation.advance.assert_not_called()
        self.assertFalse(self.app.panel.pending_reset)

    def test_running_single_step_is_ignored_for_keys_and_buttons(self):
        self.gui.button.side_effect = lambda label: label == "Single step (N)"
        self.run_frames([[event("n")]])
        self.simulation.step.assert_not_called()
        self.simulation.advance.assert_called_once()

    def test_pending_structural_edits_only_apply_on_reset_and_seed_advances(self):
        requested = self.settings.with_updates(water_depth=1.5, city_seed=42)
        self.app.controls.settings = requested
        self.run_frames([[], [event("r")], [event("c")]])
        self.simulation.advance.assert_any_call(0.0, requested)
        self.assertEqual(self.simulation.reset.call_args_list[0].args, (requested,))
        self.assertEqual(self.simulation.applied_settings.city_seed, 43)
        self.assertEqual(self.app.controls.settings.city_seed, 43)
        self.assertFalse(self.app.panel.pending_reset)
        self.assertEqual(self.renderer.update_world.call_count, 2)

    def test_overflow_pauses_physics_and_only_reset_allows_resume(self):
        error = SpatialGridOverflowError(7)
        self.simulation.advance.side_effect = [error, 0]
        self.run_frames([[], [event(" "), event("n")], [event("r")],
                         [event(" ", False), event(" ")]])
        self.assertEqual(self.simulation.advance.call_count, 2)
        self.simulation.step.assert_not_called()
        self.simulation.reset.assert_called_once_with(self.settings)
        self.assertIsNone(self.app.controls.last_error)
        self.assertFalse(self.simulation.paused)
        self.assertEqual(self.window.show.call_count, 4)

    def test_failed_reset_keeps_scene_and_ui_available(self):
        requested = self.settings.with_updates(water_depth=4)
        self.app.controls.settings = requested
        self.simulation.reset.side_effect = ValueError("capacity")
        self.run_frames([[event("r")]])
        self.assertEqual(self.simulation.applied_settings, self.settings)
        self.assertTrue(self.app.panel.pending_reset)
        self.renderer.update_world.assert_not_called()
        self.renderer.render.assert_called_once()
        self.assertIsInstance(self.app.controls.last_error, ValueError)

    def test_escape_in_same_batch_as_overflow_exits_without_presenting(self):
        self.simulation.paused = True
        self.simulation.step.side_effect = SpatialGridOverflowError(2)
        self.run_frames([[event("n"), event("Escape")]])
        self.assertTrue(self.app.controls.quit_requested)
        self.window.show.assert_not_called()

    def test_initial_render_compilation_is_excluded_from_playback_and_fps(self):
        self.set_event_batches([[], [], [], [event("Escape")]])
        # First rendering takes 30 seconds; subsequent frames take 20 ms.
        with (patch.object(self.app, "warm_up"),
              patch("fluids.app.monotonic", side_effect=[100, 130, 130, 130.02,
                                                        130.02, 130.04, 130.04])):
            self.app.run()
        durations = [call.args[0] for call in self.simulation.advance.call_args_list]
        np.testing.assert_allclose(durations, [0, 0, 0.02])
        self.assertAlmostEqual(self.app.panel.fps, 50)

    def test_warm_up_overflow_is_restored_but_remains_recoverable(self):
        error = SpatialGridOverflowError(3)
        self.simulation.step.side_effect = error
        with patch.object(ti, "sync"):
            self.app.warm_up()
        self.simulation.reset.assert_called_once_with(self.settings)
        self.assertTrue(self.simulation.paused)
        self.assertIs(self.app.controls.last_error, error)
        self.app.controls.toggle_pause()
        self.assertTrue(self.simulation.paused)


class WarmUpTests(unittest.TestCase):
    def test_real_warm_up_restores_fields_timing_and_pause_selection(self):
        ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
        self.addCleanup(ti.reset)
        settings = Settings(quality="low", water_depth=0.5, city_density=0.5)
        simulation = Simulation(settings, capacity=10000)
        positions = simulation.particles.positions.to_numpy()
        with patch.object(ti.ui, "Camera"):
            app = Application(Mock(), simulation=simulation, renderer=Mock())
        for paused in (False, True):
            simulation.paused = paused
            app.warm_up()
            self.assertEqual(simulation.paused, paused)
            self.assertEqual(simulation.simulated_time, 0)
            self.assertEqual(simulation.discarded_time, 0)
            self.assertEqual(simulation.accumulated_time, 0)
            np.testing.assert_array_equal(simulation.particles.positions.to_numpy(), positions)
            np.testing.assert_array_equal(simulation.particles.velocities.to_numpy(), 0)
            self.assertIsNone(app.controls.last_error)
