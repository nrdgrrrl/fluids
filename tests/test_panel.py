"""Panel acceptance checks without a native window or GPU allocation."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fluids.config import SETTINGS_SPECS, Settings
from fluids.controls import ACTIONS, MOUSE_HELP, Controls
from fluids.panel import BUTTON_ACTIONS, TABS, ParameterPanel


class Gui:
    """Immediate-mode input with explicit clicks/edits and recorded output."""

    def __init__(self, *, clicks=(), edits=None, tab=None):
        self.clicks = set(clicks)
        self.edits = edits or {}
        self.tab = tab
        self.lines = []
        self.sliders = []

    def sub_window(self, *args):
        return nullcontext(self)

    def text(self, text):
        self.lines.append(text)

    def button(self, label):
        return label in self.clicks

    def checkbox(self, label, selected):
        return label == self.tab if self.tab is not None else selected

    def slider_int(self, label, value, minimum, maximum):
        self.sliders.append((label, minimum, maximum, "int"))
        result = self.edits.get(label, value)
        assert minimum <= result <= maximum
        return result

    def slider_float(self, label, value, minimum, maximum):
        self.sliders.append((label, minimum, maximum, "float"))
        result = self.edits.get(label, value)
        assert minimum <= result <= maximum
        return result


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.simulation = Mock(
            applied_settings=self.settings, paused=True, last_error=None,
            particles=SimpleNamespace(active_count=1234), simulated_time=1.25,
            last_substeps=3, discarded_time=0.125,
        )
        self.simulation.requires_reset.side_effect = (
            lambda settings: settings.reset_signature() != self.simulation.applied_settings.reset_signature()
        )

        def reset(settings):
            self.simulation.applied_settings = settings
            self.simulation.last_error = None

        self.simulation.reset.side_effect = reset
        self.renderer = Mock()
        self.controls = Controls(self.simulation, renderer=self.renderer)
        self.panel = ParameterPanel(self.controls)

    def test_tabs_expose_every_spec_once_with_its_range(self):
        exposed = []
        for tab in TABS:
            gui = Gui(tab=tab)
            self.panel.draw(gui)
            self.assertEqual(self.panel.active_tab, tab)
            for name, spec in SETTINGS_SPECS.items():
                present = any(f"##{name}" in slider[0] for slider in gui.sliders)
                expected = spec.group in TABS[tab]
                self.assertEqual(present, expected, name)
                if present:
                    exposed.append(name)
                    self.assertTrue(any(spec.label in line for line in gui.lines), name)
                    if spec.requires_reset:
                        self.assertTrue(any(spec.label in line and "[reset]" in line for line in gui.lines))
                    if not spec.choices and name != "city_seed":
                        self.assertIn((f"##{name}", spec.minimum, spec.maximum,
                                       "int" if spec.integer else "float"), gui.sliders)
        self.assertCountEqual(exposed, SETTINGS_SPECS)

    def test_every_setting_accepts_endpoints_and_all_choices(self):
        for name, spec in SETTINGS_SPECS.items():
            for value in spec.choices or (spec.minimum, spec.maximum):
                with self.subTest(name=name, value=value):
                    self.assertTrue(self.panel.edit_setting(name, value))
                    self.assertEqual(getattr(self.controls.settings, name), value)
        self.simulation.reset.assert_not_called()

    def test_full_seed_range_uses_integer_widgets_without_rounding(self):
        for seed in (0, 9999, 10000, 2**31 + 1, 2**32 - 1):
            with self.subTest(seed=seed):
                gui = Gui(tab="City", edits={"Leading digits##city_seed": seed // 10000,
                                             "Last 4 digits##city_seed": seed % 10000})
                self.panel.draw(gui)
                self.assertEqual(self.controls.settings.city_seed, seed)
                for _, minimum, maximum, kind in gui.sliders:
                    if kind == "int":
                        self.assertLessEqual(maximum, 2**31 - 1)
                        self.assertGreaterEqual(minimum, 0)

    def test_pending_edits_remain_until_reset_and_reverting_clears_status(self):
        self.panel.draw(Gui(edits={"##water_depth": 4}))
        self.assertTrue(self.panel.pending_reset)
        self.assertEqual(self.simulation.applied_settings.water_depth, 2)
        self.simulation.reset.assert_not_called()
        gui = Gui()
        self.panel.draw(gui)
        self.assertIn("Pending reset: apply/reset to rebuild", gui.lines)
        self.assertIn("Water depth (m) [reset] *", gui.lines)
        self.panel.edit_setting("water_depth", 2)
        self.assertFalse(self.panel.pending_reset)
        self.panel.edit_setting("water_depth", 4)
        self.panel.draw(Gui(clicks={"Apply / reset (R)"}))
        self.assertFalse(self.panel.pending_reset)
        self.simulation.reset.assert_called_once_with(self.controls.settings)
        self.renderer.update_world.assert_called_once_with(self.simulation.world)

    def test_live_settings_change_requested_snapshot_without_reset(self):
        gui = Gui(tab="Physics", edits={"##gravity": 20, "##viscosity": 1,
                                        "##playback_speed": 4})
        self.panel.draw(gui)
        self.assertEqual(self.controls.settings.gravity, 20)
        self.assertEqual(self.controls.settings.viscosity, 1)
        self.assertEqual(self.controls.settings.playback_speed, 4)
        self.assertFalse(self.panel.pending_reset)
        self.simulation.reset.assert_not_called()

    def test_height_bounds_can_cross_and_invalid_edits_are_atomic(self):
        self.panel.draw(Gui(tab="City", edits={"##building_min_height": 10}))
        self.assertEqual(self.controls.settings.building_min_height, 10)
        self.assertEqual(self.controls.settings.building_max_height, 10)
        self.panel.draw(Gui(tab="City", edits={"##building_max_height": 0.5}))
        self.assertEqual(self.controls.settings.building_min_height, 0.5)
        self.assertEqual(self.controls.settings.building_max_height, 0.5)
        settings = self.controls.settings
        self.assertFalse(self.panel.edit_setting("gravity", 21))
        self.assertIs(self.controls.settings, settings)
        gui = Gui()
        self.panel.draw(gui)
        self.assertIn("Setting error: gravity must be at most 20.0", gui.lines)
        self.assertTrue(self.panel.edit_setting("gravity", 20))
        self.assertIsNone(self.panel.setting_error)

    def test_buttons_and_keys_dispatch_the_same_actions(self):
        for name, label in BUTTON_ACTIONS:
            action = next(action for action in ACTIONS if action.name == name)
            with self.subTest(name=name), patch.object(self.controls, name) as handler:
                button_label = "Resume" if name == "toggle_pause" else label
                self.panel.draw(Gui(clicks={f"{button_label} ({action.shortcut})"}))
                self.controls.handle_key(action.keys[0], True)
                self.assertEqual(handler.call_count, 2)
                self.controls.handle_key(action.keys[0], False)

    def test_help_lists_all_shortcuts_and_mouse_navigation_in_its_own_view(self):
        gui = Gui(clicks={"Shortcut help (H)"})
        self.panel.draw(gui)
        text = " ".join(gui.lines)
        self.assertTrue(self.controls.show_help)
        self.assertEqual(gui.sliders, [])
        for description in MOUSE_HELP:
            self.assertIn(description, text)
        for action in ACTIONS:
            self.assertIn(f"{action.shortcut}: {action.description}", text)
        self.panel.draw(Gui(clicks={"Back to settings (H)"}))
        self.assertFalse(self.controls.show_help)

    def test_diagnostics_and_errors_fit_the_panel_text_width(self):
        from fluids.simulation import SpatialGridOverflowError

        self.simulation.last_error = SpatialGridOverflowError(3)
        gui = Gui()
        self.panel.draw(gui, 0.02)
        self.assertIn("FPS: 50.0 | Particles: 1,234", gui.lines)
        self.assertIn("Simulation time: 1.250 s", gui.lines)
        self.assertIn("Substeps: 3", gui.lines)
        self.assertIn("Discarded time: 0.125 s", gui.lines)
        self.assertIn(str(self.simulation.last_error), " ".join(gui.lines))
        self.assertTrue(all(len(line) <= self.panel.TEXT_COLUMNS for line in gui.lines))

    def test_failed_reset_and_step_are_recoverable_and_reset_clears_errors(self):
        from fluids.simulation import SpatialGridOverflowError

        self.panel.edit_setting("water_depth", 4)
        settings = self.controls.settings
        self.simulation.reset.side_effect = ValueError("Initial water exceeds capacity")
        self.panel.draw(Gui(clicks={"Apply / reset (R)"}))
        self.assertIs(self.controls.settings, settings)
        self.assertTrue(self.panel.pending_reset)
        self.renderer.update_world.assert_not_called()
        error = self.controls.last_error
        self.panel.perform_action("toggle_pause")
        self.assertIs(self.controls.last_error, error)

        self.simulation.paused = True
        self.simulation.step.side_effect = SpatialGridOverflowError(7)
        self.panel.draw(Gui(clicks={"Single step (N)"}))
        self.assertTrue(self.simulation.paused)
        self.assertIs(self.controls.last_error, self.simulation.step.side_effect)
        self.simulation.reset.side_effect = lambda settings: setattr(self.simulation, "applied_settings", settings)
        self.controls.handle_key("r", True)
        self.assertIsNone(self.controls.last_error)
        self.assertFalse(self.panel.pending_reset)


if __name__ == "__main__":
    unittest.main()
