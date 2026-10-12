"""Settings, actions, diagnostics, and a separate shortcut view for GGUI.

Construct ``ParameterPanel(controls)`` once and call ``draw(gui, frame_time)``
each frame, including while paused. No GPU resources are allocated on import.
Settings are requested snapshots: only the shared reset actions rebuild a scene.
"""

from math import isfinite
from collections import deque
from textwrap import wrap

from .config import SETTINGS_SPECS
from .controls import ACTIONS, ACTION_NAMES, MOUSE_HELP


# Group every specification into one of the three user-facing tabs. Wave
# controls belong with water; playback belongs with the live physics controls.
TABS = {
    "Water": ("water", "waves"),
    "City": ("city",),
    "Physics": ("physics", "simulation"),
}
BUTTON_ACTIONS = (
    ("toggle_pause", "Pause"),
    ("single_step", "Single step"),
    ("launch_wave", "Launch wave"),
    ("reset_simulation", "Apply / reset"),
    ("new_city", "Randomize city"),
    ("toggle_help", "Shortcut help"),
)


class ParameterPanel:
    """A compact view over Controls and the shared settings specifications."""

    TEXT_COLUMNS = 48
    SEED_PART_BASE = 10_000

    def __init__(self, controls) -> None:
        self.controls = controls
        self.active_tab = "Water"
        self.setting_error: str | None = None
        self.fps = 0.0
        sim = controls.simulation
        self._timing_previous = (sim.simulated_time, sim.discarded_time)
        self._timing_samples = deque()
        self.simulation_rate = self.discard_rate = 0.0

    def _sample_timing(self, frame_time: float) -> None:
        """Rates from actual counter deltas, including pause and single-step.

        Keep roughly one wall second; reset counters start a fresh sample.
        No physics or GPU reads are needed to report slower playback honestly.
        """
        sim = self.controls.simulation
        current = (sim.simulated_time, sim.discarded_time)
        previous = self._timing_previous
        self._timing_previous = current
        if any(now < before for now, before in zip(current, previous)):
            self._timing_samples.clear()
            self.simulation_rate = self.discard_rate = 0.0
            return
        if not isfinite(frame_time) or frame_time <= 0:
            return
        self._timing_samples.append((frame_time, current[0] - previous[0],
                                     current[1] - previous[1]))
        duration = sum(sample[0] for sample in self._timing_samples)
        while len(self._timing_samples) > 1 and duration - self._timing_samples[0][0] >= 1:
            duration -= self._timing_samples.popleft()[0]
        self.simulation_rate = sum(sample[1] for sample in self._timing_samples) / duration
        self.discard_rate = sum(sample[2] for sample in self._timing_samples) / duration

    @property
    def pending_reset(self) -> bool:
        return self.controls.simulation.requires_reset(self.controls.settings)

    def _text(self, gui, message: str) -> None:
        # GGUI text does not wrap. Explicit lines also keep errors and long
        # shortcut descriptions inside the panel instead of clipping them.
        for line in message.splitlines():
            for part in wrap(line, self.TEXT_COLUMNS) or [""]:
                gui.text(part)

    def edit_setting(self, name: str, value) -> bool:
        """Validate a requested edit atomically; leave the scene untouched.

        Moving either height bound past the other moves the other bound too,
        so both controls remain usable over their entire specified ranges.
        """
        settings = self.controls.settings
        changes = {name: value}
        if name == "building_min_height" and value > settings.building_max_height:
            changes["building_max_height"] = value
        elif name == "building_max_height" and value < settings.building_min_height:
            changes["building_min_height"] = value
        try:
            updated = settings.with_updates(**changes)
        except ValueError as exc:
            self.setting_error = str(exc)
            return False
        self.controls.settings = updated
        self.setting_error = None
        return True

    def perform_action(self, name: str) -> None:
        """Use keyboard dispatch and keep recoverable failures in the panel."""
        from .simulation import SpatialGridOverflowError

        try:
            self.controls.dispatch_action(name)
        except (ValueError, SpatialGridOverflowError):
            # Dispatch records the error and pauses an overflowing solver.
            # The rest of the UI remains available to repair/reset the scene.
            pass

    def _draw_setting(self, gui, name: str, spec) -> None:
        value = getattr(self.controls.settings, name)
        label = spec.label
        if spec.unit:
            label += f" ({spec.unit})"
        if spec.requires_reset:
            label += " [reset]"
            if value != getattr(self.controls.simulation.applied_settings, name):
                label += " *"
        self._text(gui, label)
        # Put labels on their own line: ImGui otherwise places long labels to
        # the right of the slider, where they are clipped at the default width.
        widget = f"##{name}"
        if spec.choices:
            self._text(gui, str(value).capitalize())
            index = gui.slider_int(widget, spec.choices.index(value), 0, len(spec.choices) - 1)
            # ImGui's text-entry mode can supply values outside slider bounds.
            index = max(0, min(len(spec.choices) - 1, index))
            edited = spec.choices[index]
        elif spec.integer and spec.maximum > 2**31 - 1:
            # ImGui SliderInt accepts signed 32-bit values. Two decimal parts
            # retain the full uint32 seed range without float rounding.
            self._text(gui, str(value))
            base = self.SEED_PART_BASE
            upper = gui.slider_int(f"Leading digits##{name}", value // base,
                                   spec.minimum // base, spec.maximum // base)
            lower_min = spec.minimum % base if upper == spec.minimum // base else 0
            lower_max = spec.maximum % base if upper == spec.maximum // base else base - 1
            lower = gui.slider_int(f"Last 4 digits##{name}", min(lower_max, max(lower_min, value % base)),
                                   lower_min, lower_max)
            edited = upper * base + lower
        elif spec.integer:
            edited = gui.slider_int(widget, value, spec.minimum, spec.maximum)
        else:
            edited = gui.slider_float(widget, value, spec.minimum, spec.maximum)
        if edited != value:
            self.edit_setting(name, edited)

    def _draw_help(self, gui) -> None:
        gui.text("Mouse navigation")
        for description in MOUSE_HELP:
            self._text(gui, description)
        gui.text("Action shortcuts")
        for action in ACTIONS:
            self._text(gui, f"{action.shortcut}: {action.description}")

    def draw(self, gui, frame_time: float = 0.0) -> None:
        """Draw one view within the same rectangle used for mouse capture."""
        self._sample_timing(frame_time)
        if isfinite(frame_time) and frame_time > 0:
            instant_fps = 1.0 / frame_time
            self.fps = instant_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instant_fps
        controls = self.controls
        simulation = controls.simulation
        with controls.panel(gui):
            for name, label in BUTTON_ACTIONS:
                if name == "toggle_pause" and simulation.paused:
                    label = "Resume"
                elif name == "toggle_help" and controls.show_help:
                    label = "Back to settings"
                if gui.button(f"{label} ({ACTION_NAMES[name].shortcut})"):
                    self.perform_action(name)

            state = "Paused" if simulation.paused else "Running"
            if not simulation.paused and controls.settings.playback_speed == 0:
                state = "Stopped (playback speed is zero)"
            gui.text(state)
            gui.text(f"FPS: {self.fps:.1f} | Particles: {simulation.particles.active_count:,}")
            gui.text(f"Simulation time: {simulation.simulated_time:.3f} s")
            gui.text(f"Substeps: {simulation.last_substeps}")
            gui.text(f"Discarded time: {simulation.discarded_time:.3f} s")
            gui.text(f"Sim rate: {self.simulation_rate:.3f} s/wall s")
            gui.text(f"Discard rate: {self.discard_rate:.3f} s/wall s")
            gui.text(f"Requested speed: {controls.settings.playback_speed:.2f}x")
            gui.text("Pending reset: apply/reset to rebuild" if self.pending_reset
                     else "No pending reset")

            error = controls.last_error or simulation.last_error
            if error is not None:
                self._text(gui, f"Simulation error: {error}")
            if self.setting_error:
                self._text(gui, f"Setting error: {self.setting_error}")

            if controls.show_help:
                self._draw_help(gui)
            else:
                # GGUI has no tab widget. Exclusive selectors provide tabs
                # using its public API, without depending on private ImGui.
                for tab in TABS:
                    if gui.checkbox(tab, self.active_tab == tab):
                        self.active_tab = tab
                gui.text("[reset] applies on reset; * marks an edit")
                if self.active_tab == "Water":
                    self._text(gui, "Legacy amplitude/period/direction do not affect launched waves.")
                elif self.active_tab == "City":
                    self._text(gui, "Height bounds move together if crossed.")
                for name, spec in SETTINGS_SPECS.items():
                    if spec.group in TABS[self.active_tab]:
                        self._draw_setting(gui, name, spec)
