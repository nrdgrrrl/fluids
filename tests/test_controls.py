"""Camera geometry, UI gesture capture, and discrete action acceptance checks."""

import os
import unittest
from contextlib import nullcontext
from math import cos, dist, isfinite, sin
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fluids.config import Settings
from fluids.controls import ACTIONS, PANEL_RECT, CameraController, Controls, PanelRect, main, run_scene


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraController()

    def view(self):
        camera = self.camera
        return camera.target, camera.azimuth, camera.elevation, camera.distance

    def drag(self, cursor, *, left=False, right=False, **kwargs):
        self.camera.update_mouse(cursor, left_pressed=left, right_pressed=right, **kwargs)

    def test_continuous_orbit_keeps_distance_and_never_flips(self):
        target = self.camera.target
        self.drag((0.5, 0.5), left=True)
        azimuth = self.camera.azimuth
        for index in range(1, 200):
            self.drag((0.5 + index * 0.02, 0.5 + index * 0.02), left=True)
            self.assertGreaterEqual(self.camera.elevation, self.camera.MIN_ELEVATION)
            self.assertLessEqual(self.camera.elevation, self.camera.MAX_ELEVATION)
            self.assertGreater(self.camera.position[1], target[1])
            self.assertAlmostEqual(dist(self.camera.position, target), self.camera.distance)
        self.assertNotEqual(azimuth, self.camera.azimuth)
        self.assertEqual(target, self.camera.target)
        self.camera.orbit(0, -1e6)
        self.assertEqual(self.camera.elevation, self.camera.MAX_ELEVATION)
        self.assertTrue(all(isfinite(value) for value in self.camera.position))

    def test_zoom_clamps_both_ends_and_reset_restores_entire_view(self):
        default = self.view()
        self.camera.zoom(1000000)
        self.assertEqual(self.camera.distance, self.camera.MIN_DISTANCE)
        self.camera.zoom(-1000000)
        self.assertEqual(self.camera.distance, self.camera.MAX_DISTANCE)
        self.camera.pan(0.2, -0.2)
        self.camera.orbit(0.1, 0.1)
        self.camera.reset()
        self.assertEqual(default, self.view())

    def test_pan_translates_eye_and_target_without_changing_orientation(self):
        before_position, before_target = self.camera.position, self.camera.target
        before_angles = self.camera.azimuth, self.camera.elevation
        self.drag((0.5, 0.5), right=True)
        self.drag((0.6, 0.6), right=True, aspect_ratio=1.6)
        self.assertNotEqual(self.camera.target, before_target)
        for eye, target, old_eye, old_target in zip(
            self.camera.position, self.camera.target, before_position, before_target,
        ):
            self.assertAlmostEqual(eye - old_eye, target - old_target)
        self.assertEqual(before_angles, (self.camera.azimuth, self.camera.elevation))

    def test_horizontal_pan_uses_current_view_and_aspect_ratio(self):
        for azimuth in (0, 1.7, 3.14):
            self.camera.reset()
            self.camera.azimuth = azimuth
            target = self.camera.target
            self.camera.pan(0.1, 0, 2)
            delta = tuple(new - old for new, old in zip(self.camera.target, target))
            self.assertAlmostEqual(delta[1], 0)
            self.assertAlmostEqual(delta[0] * sin(azimuth) + delta[2] * cos(azimuth), 0)
            double_width = dist(self.camera.target, target)
            self.camera.target = target
            self.camera.pan(0.1, 0, 1)
            self.assertAlmostEqual(double_width, 2 * dist(self.camera.target, target))

    def test_panel_drag_stays_excluded_outside_panel_until_release(self):
        for button in ("left", "right"):
            with self.subTest(button=button):
                self.camera.reset()
                before = self.view()
                self.drag((0.1, 0.8), **{button: True})
                self.drag((0.8, 0.2), **{button: True})
                self.drag((0.9, 0.9), left=True, right=True)
                self.assertEqual(before, self.view())
                self.drag((0.9, 0.9))
                self.drag((0.7, 0.5), **{button: True})
                self.drag((0.8, 0.6), **{button: True})
                self.assertNotEqual(before, self.view())
                self.drag((0.8, 0.6))

    def test_scene_drag_can_cross_panel_and_new_drag_has_no_jump(self):
        self.drag((0.7, 0.5), left=True)
        self.drag((0.1, 0.8), left=True)
        self.assertNotEqual(self.camera.azimuth, self.camera.DEFAULT_AZIMUTH)
        self.drag((0.1, 0.8))
        before = self.view()
        self.drag((0.9, 0.1), left=True)
        self.assertEqual(before, self.view())

    def test_reset_while_dragging_slider_does_not_release_capture(self):
        self.drag((0.1, 0.8), left=True)
        self.camera.reset()
        before = self.view()
        self.drag((0.8, 0.2), left=True)
        self.assertEqual(before, self.view())

    def test_panel_placement_matches_cursor_hit_testing(self):
        gui = Mock()
        rect = PanelRect(0.1, 0.2, 0.3, 0.4)
        rect.sub_window(gui)
        gui.sub_window.assert_called_once_with("Controls", 0.1, 0.2, 0.3, 0.4)
        self.assertTrue(rect.contains((0.2, 0.7)))
        self.assertFalse(rect.contains((0.2, 0.9)))
        self.assertFalse(rect.contains((0.5, 0.7)))

    def test_apply_uses_y_up_camera_and_perspective(self):
        camera = Mock()
        self.camera.apply(camera)
        camera.position.assert_called_once_with(*self.camera.position)
        camera.lookat.assert_called_once_with(*self.camera.target)
        camera.up.assert_called_once_with(0, 1, 0)
        camera.fov.assert_called_once_with(self.camera.FOV)


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(quality="low")
        self.simulation = Mock(applied_settings=self.settings, paused=True)
        self.renderer = Mock()
        self.controls = Controls(self.simulation, renderer=self.renderer)
        self.window = Mock()
        self.window.get_events.return_value = []
        self.window.get_window_shape.return_value = (1440, 900)
        self.window.get_cursor_pos.return_value = (0.7, 0.5)
        self.window.is_pressed.return_value = False

    def test_documented_keys_dispatch_and_help_comes_from_action_metadata(self):
        for action in ACTIONS:
            with self.subTest(action=action.name), patch.object(self.controls, action.name) as handler:
                self.assertEqual(self.controls.handle_key(action.keys[0], True), action.name)
                handler.assert_called_once_with()
        gui = Mock()
        self.controls.show_help = True
        self.controls.draw_help(gui)
        help_lines = [call.args[0] for call in gui.text.call_args_list]
        for action in ACTIONS:
            self.assertIn(f"{action.shortcut}: {action.description}", help_lines)

    def test_holding_wave_key_never_launches_per_frame_or_on_repeat(self):
        press = SimpleNamespace(key="w", action="Press")
        self.window.get_events.return_value = [press]
        for _ in range(50):
            self.controls.update(self.window)
        self.simulation.launch_wave.assert_called_once_with(self.settings)
        self.window.get_events.return_value = [SimpleNamespace(key="w", action="Release"), press]
        self.controls.update(self.window)
        self.assertEqual(self.simulation.launch_wave.call_count, 2)

    def test_fast_press_and_release_between_frames_is_dispatched(self):
        self.window.get_events.return_value = [
            SimpleNamespace(key="w", action="Press"),
            SimpleNamespace(key="w", action="Release"),
            SimpleNamespace(key="w", action="Press"),
            SimpleNamespace(key="w", action="Release"),
        ]
        self.controls.update(self.window)
        self.assertEqual(self.simulation.launch_wave.call_count, 2)

    def test_all_discrete_actions_are_edge_triggered_and_case_insensitive(self):
        for action in ACTIONS:
            with self.subTest(action=action.name), patch.object(self.controls, action.name) as handler:
                key = action.keys[0].upper()
                self.controls.handle_key(key, True)
                self.controls.handle_key(key, True)
                handler.assert_called_once_with()
                self.controls.handle_key(key, False)
                self.controls.handle_key(key, True)
                self.assertEqual(handler.call_count, 2)

    def test_solver_error_preserves_queued_releases_and_next_press_after_reset(self):
        from fluids.simulation import SpatialGridOverflowError

        self.controls.handle_key("w", True)
        error = SpatialGridOverflowError(3)
        self.simulation.step.side_effect = error
        self.window.get_events.return_value = [
            SimpleNamespace(key="n", action="Press"),
            SimpleNamespace(key="N", action="Release"),
            SimpleNamespace(key="w", action="Release"),
            SimpleNamespace(key="c", action="Press"),
            SimpleNamespace(key="c", action="Release"),
        ]
        with self.assertRaises(SpatialGridOverflowError) as raised:
            self.controls.update(self.window)
        self.assertIs(raised.exception, error)
        self.simulation.step.assert_called_once_with(self.settings)
        self.simulation.reset.assert_not_called()

        self.window.get_events.return_value = [
            SimpleNamespace(key="r", action="Press"),
            SimpleNamespace(key="r", action="Release"),
        ]
        self.controls.update(self.window)
        self.simulation.reset.assert_called_once_with(self.settings)
        self.simulation.step.side_effect = None
        self.window.get_events.return_value = [
            SimpleNamespace(key="n", action="Press"),
            SimpleNamespace(key="w", action="Press"),
        ]
        self.controls.update(self.window)
        self.assertEqual(self.simulation.step.call_count, 2)
        self.assertEqual(self.simulation.launch_wave.call_count, 2)
        self.controls.update(self.window)
        self.assertEqual(self.simulation.step.call_count, 2)
        self.assertEqual(self.simulation.launch_wave.call_count, 2)

    def test_paused_camera_and_zoom_work_without_advancing_physics(self):
        self.window.is_pressed.side_effect = lambda key: key == "RMB"
        self.controls.update(self.window)
        target = self.controls.camera.target
        self.window.get_cursor_pos.return_value = (0.8, 0.6)
        self.window.get_events.return_value = [SimpleNamespace(key="=", action="Press")]
        self.controls.update(self.window)
        self.assertNotEqual(target, self.controls.camera.target)
        self.assertLess(self.controls.camera.distance, self.controls.camera.DEFAULT_DISTANCE)
        self.assertTrue(self.simulation.paused)
        self.simulation.advance.assert_not_called()
        self.simulation.step.assert_not_called()
        self.controls.handle_key("0", True)
        self.assertEqual(self.controls.camera.target, self.controls.camera.DEFAULT_TARGET)

    def test_simulation_actions_use_current_ui_settings_and_update_renderer(self):
        settings = self.settings.with_updates(wave_speed=15, city_density=0.2)
        self.controls.settings = settings
        self.controls.handle_key(" ", True)
        self.assertFalse(self.simulation.paused)
        self.controls.handle_key("n", True)
        self.simulation.step.assert_called_once_with(settings)
        self.controls.handle_key("w", True)
        self.simulation.launch_wave.assert_called_once_with(settings)
        self.controls.handle_key("r", True)
        self.simulation.reset.assert_called_once_with(settings)
        self.renderer.update_world.assert_called_once_with(self.simulation.world)
        self.controls.handle_key("c", True)
        new_settings = self.controls.settings
        self.assertEqual(new_settings.city_seed, settings.city_seed + 1)
        self.assertEqual(new_settings.city_density, settings.city_density)
        self.simulation.reset.assert_called_with(new_settings)
        self.assertEqual(self.renderer.update_world.call_count, 2)

    def test_failed_city_reset_keeps_settings_and_geometry_and_seed_wraps(self):
        settings = self.settings.with_updates(city_seed=2**32 - 1)
        self.controls.settings = settings
        self.simulation.reset.side_effect = ValueError("capacity")
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.controls.handle_key("c", True)
        self.assertIs(self.controls.settings, settings)
        self.renderer.update_world.assert_not_called()
        self.controls.handle_key("c", False)
        self.simulation.reset.side_effect = None
        self.controls.handle_key("c", True)
        self.assertEqual(self.controls.settings.city_seed, 0)

    def test_help_toggle_and_escape_exit(self):
        self.controls.show_help = True
        self.controls.handle_key("h", True)
        gui = Mock()
        self.controls.draw_help(gui)
        self.assertEqual(gui.text.call_count, 1)
        self.window.get_events.return_value = [SimpleNamespace(key="Escape", action="Press")]
        self.assertFalse(self.controls.update(self.window))
        self.assertFalse(self.window.running)

    def test_unknown_keys_and_motion_events_have_no_actions(self):
        self.assertIsNone(self.controls.handle_key("unknown", True))
        self.window.get_events.return_value = [SimpleNamespace(key="w", action="Motion")]
        self.controls.update(self.window)
        self.simulation.launch_wave.assert_not_called()

    def test_custom_panel_is_used_for_both_ui_and_mouse_capture(self):
        rect = PanelRect(0.6, 0.2, 0.3, 0.6)
        self.controls.panel_rect = rect
        self.window.is_pressed.side_effect = lambda key: key == "LMB"
        self.controls.update(self.window)
        self.window.get_cursor_pos.return_value = (0.1, 0.1)
        self.controls.update(self.window)
        self.assertEqual(self.controls.camera.azimuth, self.controls.camera.DEFAULT_AZIMUTH)
        gui = Mock()
        self.controls.panel(gui)
        gui.sub_window.assert_called_once_with("Controls", rect.x, rect.y, rect.width, rect.height)

    def test_release_and_new_drag_between_frames_does_not_inherit_panel_capture(self):
        self.window.is_pressed.side_effect = lambda key: key == "LMB"
        self.window.get_cursor_pos.return_value = (0.1, 0.8)
        self.window.get_events.return_value = [SimpleNamespace(key="LMB", action="Press")]
        self.controls.update(self.window)
        self.window.get_cursor_pos.return_value = (0.7, 0.5)
        self.window.get_events.return_value = [
            SimpleNamespace(key="LMB", action="Release"),
            SimpleNamespace(key="LMB", action="Press"),
        ]
        self.controls.update(self.window)
        self.window.get_events.return_value = []
        self.window.get_cursor_pos.return_value = (0.8, 0.6)
        self.controls.update(self.window)
        self.assertNotEqual(self.controls.camera.azimuth, self.controls.camera.DEFAULT_AZIMUTH)

    def test_scene_loop_handles_pause_sliders_rendering_and_escape(self):
        import taichi as ti

        self.simulation.last_error = None
        self.simulation.particles.active_count = 100
        self.simulation.simulated_time = self.simulation.discarded_time = 0.0
        self.simulation.last_substeps = 0
        self.simulation.requires_reset.return_value = False
        self.window.running = True
        self.window.get_events.side_effect = [
            [], [], [SimpleNamespace(key="Escape", action="Press")],
        ]
        gui = self.window.get_gui.return_value
        gui.sub_window.side_effect = lambda *args: nullcontext(gui)
        gui.button.return_value = False
        gui.checkbox.side_effect = lambda label, selected: selected
        gui.slider_int.side_effect = lambda label, value, minimum, maximum: value
        gui.slider_float.side_effect = lambda label, value, minimum, maximum: value
        camera = Mock()
        with patch.object(ti.ui, "Camera", return_value=camera):
            controls = run_scene(self.window, self.simulation, self.renderer)
        self.assertTrue(controls.quit_requested)
        self.assertEqual(self.window.show.call_count, 2)
        self.assertEqual(camera.position.call_count, 2)
        self.assertEqual(self.renderer.render.call_count, 2)
        self.assertTrue(self.simulation.paused)
        gui.sub_window.assert_called_with(
            "Controls", PANEL_RECT.x, PANEL_RECT.y, PANEL_RECT.width, PANEL_RECT.height,
        )
        self.assertEqual(gui.slider_float.call_count, 14)

    def test_scene_loop_pauses_on_solver_error_and_camera_still_updates(self):
        import taichi as ti
        from fluids.simulation import SpatialGridOverflowError

        error = SpatialGridOverflowError(3)
        self.simulation.advance.side_effect = error
        self.simulation.last_error = error
        self.simulation.particles.active_count = 100
        self.simulation.simulated_time = self.simulation.discarded_time = 0.0
        self.simulation.last_substeps = 0
        self.simulation.requires_reset.return_value = False
        self.window.running = True
        self.window.get_events.side_effect = [
            [], [SimpleNamespace(key="Escape", action="Press")],
        ]
        gui = self.window.get_gui.return_value
        gui.sub_window.side_effect = lambda *args: nullcontext(gui)
        gui.button.return_value = False
        gui.checkbox.side_effect = lambda label, selected: selected
        gui.slider_int.side_effect = lambda label, value, minimum, maximum: value
        gui.slider_float.side_effect = lambda label, value, minimum, maximum: value
        camera = Mock()
        with patch.object(ti.ui, "Camera", return_value=camera):
            run_scene(self.window, self.simulation, self.renderer)
        self.assertTrue(self.simulation.paused)
        text = " ".join(call.args[0] for call in gui.text.call_args_list)
        self.assertIn(str(error), text)
        camera.position.assert_called_once()
        self.window.show.assert_called_once()

    def test_scene_startup_always_destroys_window_on_interrupt(self):
        import taichi as ti
        from fluids import bootstrap

        with (
            patch.dict(os.environ, {"DISPLAY": ":0"}),
            patch.object(bootstrap, "initialize_gpu"),
            patch.object(ti.ui, "Window", return_value=self.window),
            patch("fluids.simulation.Simulation", return_value=self.simulation),
            patch("fluids.rendering.Renderer", return_value=self.renderer),
            patch("fluids.controls.run_scene", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(main(), 130)
        self.window.destroy.assert_called_once_with()


class SimulationControlTests(unittest.TestCase):
    def test_paused_actions_work_with_real_simulation(self):
        import numpy as np
        import taichi as ti
        from fluids.simulation import Simulation

        ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=False)
        self.addCleanup(ti.reset)
        settings = Settings(quality="low", water_depth=0.5, city_density=0)
        simulation = Simulation(settings, capacity=10000)
        simulation.paused = True
        controls = Controls(simulation)
        count = simulation.particles.active_count
        positions = simulation.particles.positions.to_numpy()
        controls.handle_key("w", True)
        self.assertEqual(simulation.simulated_time, 0)
        np.testing.assert_array_equal(simulation.particles.positions.to_numpy(), positions)
        self.assertGreater(np.linalg.norm(simulation.particles.velocities.to_numpy()), 0)
        controls.handle_key("n", True)
        self.assertGreater(simulation.simulated_time, 0)
        self.assertTrue(simulation.paused)
        controls.handle_key("r", True)
        self.assertEqual(simulation.simulated_time, 0)
        self.assertEqual(simulation.particles.active_count, count)
        np.testing.assert_array_equal(simulation.particles.positions.to_numpy(), positions)
        controls.handle_key("c", True)
        self.assertEqual(simulation.applied_settings.city_seed, settings.city_seed + 1)
        self.assertTrue(simulation.paused)


if __name__ == "__main__":
    unittest.main()
