"""Mouse camera controls and the scene's shared keyboard action contract.

Like ``Renderer``, this component targets a Taichi GGUI scene, independently of
the resizable clear-window bootstrap checkpoint. No GPU resources are allocated
on import. Construct ``Controls(simulation)`` once, then, on *every* frame:

    controls.update(window)  # includes paused frames; consumes input events
    simulation.advance(frame_time, controls.settings)
    renderer.update_world(simulation.world)  # R and C can rebuild the city
    controls.camera.apply(camera)
    scene.set_camera(camera)
    panel.draw(window.get_gui(), frame_time)

Construct ``ParameterPanel(controls)`` from ``fluids.panel`` once before the loop.

PANEL_RECT uses GGUI's normalized top-left UI coordinates. Mouse positions use
normalized bottom-left coordinates. The same rectangle places the panel and
captures mouse gestures, including its title, padding, and scrollbar. Input
updates must not be gated on Simulation.paused or playback speed.

The desktop loop lives in ``fluids.app``. Both ``fluids`` and
``python -m fluids.controls`` launch it; GGUI in Taichi 1.7.4 has a fixed size.
"""

from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from types import MappingProxyType

from .config import DOMAIN_SIZE, SETTINGS_SPECS


@dataclass(frozen=True)
class PanelRect:
    """Normalized UI bounds, with inclusive mouse hit testing."""

    x: float
    y: float
    width: float
    height: float

    def contains(self, cursor: tuple[float, float]) -> bool:
        x, bottom_y = cursor
        y = 1.0 - bottom_y
        return (
            self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )

    def sub_window(self, gui, title: str = "Controls"):
        """Place all sliders inside exactly the bounds used for hit testing."""
        return gui.sub_window(title, self.x, self.y, self.width, self.height)


PANEL_RECT = PanelRect(0.01, 0.01, 0.28, 0.90)


@dataclass(frozen=True)
class Action:
    name: str
    keys: tuple[str, ...]
    shortcut: str
    description: str


# This is the sole source for keyboard dispatch and on-screen action help.
# GGUI reports letters in lowercase and the main keyboard's plus key as '='.
ACTIONS = (
    Action("toggle_pause", (" ",), "Space", "Pause or resume physics"),
    Action("single_step", ("n",), "N", "Advance one physics step while paused"),
    Action("reset_simulation", ("r",), "R", "Reset water and city with current settings"),
    Action("new_city", ("c",), "C", "Generate a new city and reset water"),
    Action("launch_wave", ("w",), "W", "Launch one wave with current settings"),
    Action("toggle_help", ("h",), "H", "Show or hide shortcut help"),
    Action("zoom_in", ("+", "="), "+", "Zoom in"),
    Action("zoom_out", ("-",), "-", "Zoom out"),
    Action("reset_view", ("0",), "0", "Restore the default camera view"),
    Action("quit", ("escape",), "Escape", "Exit"),
)
ACTION_KEYS = MappingProxyType({key: action for action in ACTIONS for key in action.keys})
ACTION_NAMES = MappingProxyType({action.name: action for action in ACTIONS})
MOUSE_HELP = ("Left drag: Orbit the scene", "Right drag: Pan the camera target")


class CameraController:
    """Y-up orbit camera with a fixed up vector and bounded zoom/elevation.

    Cursor deltas are normalized window distances. Panning follows the screen's
    right/up basis and is scaled to the current perspective distance and aspect
    ratio, so it remains usable after zooming. The first pressed mouse button
    owns a gesture until all buttons are released. A panel-owned gesture stays
    excluded even if another button is pressed outside the panel.
    """

    MIN_ELEVATION = radians(5.0)
    MAX_ELEVATION = radians(85.0)
    MIN_DISTANCE = 5.0
    MAX_DISTANCE = 160.0
    DEFAULT_TARGET = (0.0, 3.0, DOMAIN_SIZE[2] / 2)
    DEFAULT_AZIMUTH = radians(-55.0)
    DEFAULT_ELEVATION = radians(30.0)
    DEFAULT_DISTANCE = 55.0
    FOV = 45.0
    ZOOM_FACTOR = 1.1

    def __init__(self) -> None:
        self._gesture: str | None = None
        self._cursor: tuple[float, float] | None = None
        self.reset()

    def reset(self) -> None:
        """Restore target, orientation, and distance without breaking capture."""
        self.target = self.DEFAULT_TARGET
        self.azimuth = self.DEFAULT_AZIMUTH
        self.elevation = self.DEFAULT_ELEVATION
        self.distance = self.DEFAULT_DISTANCE

    @property
    def position(self) -> tuple[float, float, float]:
        horizontal = self.distance * cos(self.elevation)
        return (
            self.target[0] + horizontal * sin(self.azimuth),
            self.target[1] + self.distance * sin(self.elevation),
            self.target[2] + horizontal * cos(self.azimuth),
        )

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth = (self.azimuth - dx * 2 * pi) % (2 * pi)
        self.elevation = min(
            self.MAX_ELEVATION, max(self.MIN_ELEVATION, self.elevation - dy * pi),
        )

    def pan(self, dx: float, dy: float, aspect_ratio: float = 1.0) -> None:
        scale = 2 * self.distance * tan(radians(self.FOV) / 2)
        right = (cos(self.azimuth), 0.0, -sin(self.azimuth))
        up = (
            -sin(self.elevation) * sin(self.azimuth), cos(self.elevation),
            -sin(self.elevation) * cos(self.azimuth),
        )
        self.target = tuple(
            value - scale * (dx * aspect_ratio * r + dy * u)
            for value, r, u in zip(self.target, right, up)
        )

    def zoom(self, steps: int) -> None:
        """Positive steps zoom in; repeated input saturates at usable bounds."""
        # Bound the exponent too, allowing arbitrarily large step requests.
        factor = self.ZOOM_FACTOR ** max(-100, min(100, -steps))
        self.distance = min(self.MAX_DISTANCE, max(self.MIN_DISTANCE, self.distance * factor))

    def update_mouse(
        self, cursor: tuple[float, float], *, left_pressed: bool,
        right_pressed: bool, panel_rect: PanelRect = PANEL_RECT,
        aspect_ratio: float = 1.0,
    ) -> None:
        if not left_pressed and not right_pressed:
            self._gesture = None
            self._cursor = None
            return
        if self._gesture is None:
            self._gesture = (
                "panel" if panel_rect.contains(cursor)
                else "orbit" if left_pressed else "pan"
            )
            self._cursor = cursor
            return
        dx, dy = (value - previous for value, previous in zip(cursor, self._cursor))
        self._cursor = cursor
        if self._gesture == "orbit" and left_pressed:
            self.orbit(dx, dy)
        elif self._gesture == "pan" and right_pressed:
            self.pan(dx, dy, aspect_ratio)

    def apply(self, camera) -> None:
        """Write this view to a ti.ui.Camera; no built-in WASD tracking needed."""
        camera.position(*self.position)
        camera.lookat(*self.target)
        camera.up(0.0, 1.0, 0.0)
        camera.fov(self.FOV)


class Controls:
    """Scene actions and camera input, independent of physics playback.

    ``settings`` is the current requested UI snapshot; R applies pending scene
    settings, C increments the seed and applies them. Attach an optional renderer
    to synchronize reset geometry immediately, including while paused. ``update``
    returns False after Escape so the caller can leave its loop before drawing.
    """

    def __init__(
        self, simulation, *, settings=None, renderer=None,
        panel_rect: PanelRect = PANEL_RECT,
    ) -> None:
        self.simulation = simulation
        self.settings = simulation.applied_settings if settings is None else settings
        self.renderer = renderer
        self.panel_rect = panel_rect
        self.camera = CameraController()
        self.show_help = False
        self.last_error: Exception | None = None
        self.quit_requested = False
        self._held_keys: set[str] = set()
        self._mouse_buttons: set[str] = set()

    def handle_key(self, key: str, pressed: bool) -> str | None:
        """Dispatch once per physical press, ignoring repeats until release."""
        key = key.lower()
        if not pressed:
            self._held_keys.discard(key)
            return None
        action = ACTION_KEYS.get(key)
        if action is None or key in self._held_keys:
            return None
        self._held_keys.add(key)
        self.dispatch_action(action.name)
        return action.name

    def dispatch_action(self, name: str) -> None:
        """Shared button/keyboard entry point, with recoverable error state.

        Callers catch recoverable exceptions to continue drawing the scene.
        Successful reset actions clear the error; unrelated actions retain it.
        """
        from .simulation import SpatialGridOverflowError

        action = ACTION_NAMES[name]
        try:
            getattr(self, action.name)()
        except (ValueError, SpatialGridOverflowError) as exc:
            self.last_error = exc
            if isinstance(exc, SpatialGridOverflowError):
                self.simulation.paused = True
            raise
        if name in ("reset_simulation", "new_city"):
            self.last_error = None

    def update(self, window) -> bool:
        """Consume GGUI press/release events, then update continuous mouse input."""
        import taichi as ti

        # Taichi 1.7.4's native GGUI Event exposes only key, without action
        # or type. Ask the window to classify events using its public filters.
        # Releases first let a new drag start without inheriting old capture.
        events = iter([
            *((event, False) for event in window.get_events(ti.ui.RELEASE)),
            *((event, True) for event in window.get_events(ti.ui.PRESS)),
        ])
        cursor = window.get_cursor_pos()
        width, height = window.get_window_shape()
        aspect_ratio = width / max(height, 1)
        try:
            for event, pressed in events:
                key = getattr(event, "key", None)
                if not isinstance(key, str):
                    continue
                if key in ("LMB", "RMB"):
                    if pressed:
                        self._mouse_buttons.add(key)
                    else:
                        self._mouse_buttons.discard(key)
                    self.camera.update_mouse(
                        cursor, left_pressed="LMB" in self._mouse_buttons,
                        right_pressed="RMB" in self._mouse_buttons,
                        panel_rect=self.panel_rect, aspect_ratio=aspect_ratio,
                    )
                else:
                    if pressed:
                        # GGUI queues GLFW_PRESS, never GLFW_REPEAT. Every
                        # queued press is a new physical press, even when a
                        # key was tapped several times between frames.
                        self.handle_key(key, False)
                    self.handle_key(key, pressed)
        finally:
            # GGUI consumed the whole batch. If an action failed, still honor
            # Escape before the error reaches the scene loop.
            for event, pressed in events:
                key = getattr(event, "key", None)
                if not isinstance(key, str):
                    continue
                if not pressed and key not in ("LMB", "RMB"):
                    self.handle_key(key, False)
                elif pressed and key.lower() == "escape":
                    self.handle_key(key, True)
        self._mouse_buttons = {
            key for key in ("LMB", "RMB") if window.is_pressed(key)
        }
        self.camera.update_mouse(
            cursor, left_pressed="LMB" in self._mouse_buttons,
            right_pressed="RMB" in self._mouse_buttons,
            panel_rect=self.panel_rect, aspect_ratio=aspect_ratio,
        )
        if self.quit_requested:
            window.running = False
        return not self.quit_requested

    def panel(self, gui):
        return self.panel_rect.sub_window(gui)

    def draw_help(self, gui) -> None:
        """Declare help within ``panel`` using the same metadata as dispatch."""
        help_action = ACTION_KEYS["h"]
        gui.text(f"{help_action.shortcut}: {help_action.description}")
        if self.show_help:
            for description in MOUSE_HELP:
                gui.text(description)
            for action in ACTIONS:
                if action is not help_action:
                    gui.text(f"{action.shortcut}: {action.description}")

    def toggle_pause(self) -> None:
        if self._overflowed():
            return
        self.simulation.paused = not self.simulation.paused

    def single_step(self) -> None:
        if self.simulation.paused and not self._overflowed():
            self.simulation.step(self.settings)

    def _overflowed(self) -> bool:
        """Reset is required before another attempt at an incomplete grid."""
        from .simulation import SpatialGridOverflowError

        return isinstance(self.last_error, SpatialGridOverflowError) or isinstance(
            self.simulation.last_error, SpatialGridOverflowError,
        )

    def reset_simulation(self) -> None:
        self.simulation.reset(self.settings)
        if self.renderer is not None:
            self.renderer.update_world(self.simulation.world)

    def new_city(self) -> None:
        maximum = SETTINGS_SPECS["city_seed"].maximum
        settings = self.settings.with_updates(
            city_seed=(self.settings.city_seed + 1) % (maximum + 1),
        )
        self.simulation.reset(settings)
        self.settings = settings
        if self.renderer is not None:
            self.renderer.update_world(self.simulation.world)

    def launch_wave(self) -> None:
        self.simulation.launch_wave(self.settings)

    def toggle_help(self) -> None:
        self.show_help = not self.show_help

    def zoom_in(self) -> None:
        self.camera.zoom(1)

    def zoom_out(self) -> None:
        self.camera.zoom(-1)

    def reset_view(self) -> None:
        self.camera.reset()

    def quit(self) -> None:
        self.quit_requested = True


def run_scene(window, simulation, renderer) -> Controls:
    """Compatibility entry point for callers that already own scene resources."""
    from .app import Application

    return Application(window, simulation=simulation, renderer=renderer).run()


def main() -> int:
    """Launch the same application as the console entry point."""
    from .__main__ import main as app_main

    return app_main()


if __name__ == "__main__":
    import sys

    sys.exit(main())
