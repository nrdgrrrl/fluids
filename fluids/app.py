"""Desktop application connecting input, playback, diagnostics, and GGUI."""

import os
import sys
from time import monotonic

import taichi as ti

from .config import DEFAULT_SETTINGS
from .controls import Controls
from .panel import ParameterPanel
from .rendering import Renderer
from .simulation import Simulation, SpatialGridOverflowError


BACKGROUND_COLOR = (0.035, 0.055, 0.085)


def create_window():
    """Create the Vulkan scene window used by all interactive components."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Native window initialization failed. Run Fluids in a desktop graphics "
            "session with DISPLAY set (enable XWayland on Wayland desktops)."
        )
    try:
        return ti.ui.Window("Fluids — coastal wave sandbox", (1440, 900), vsync=True)
    except Exception as exc:
        raise RuntimeError(
            "Native window initialization failed. Install a compatible Vulkan GPU "
            "driver and run Fluids in a desktop graphics session with GLFW support."
        ) from exc


class Application:
    """Own one set of scene components; the caller owns the window lifetime."""

    def __init__(self, window, *, settings=DEFAULT_SETTINGS, simulation=None, renderer=None):
        self.window = window
        self.simulation = Simulation(settings) if simulation is None else simulation
        self.renderer = Renderer(self.simulation.world) if renderer is None else renderer
        self.controls = Controls(self.simulation, renderer=self.renderer)
        self.panel = ParameterPanel(self.controls)
        self.camera = ti.ui.Camera()
        self.canvas = window.get_canvas()
        self.scene = window.get_scene()
        self.gui = window.get_gui()

    def warm_up(self) -> None:
        """Compile physics and wave kernels, then restore the initial scene.

        Runtime settings are kernel arguments, so resets do not recompile these
        passes. No warm-up motion, impulse, or simulated time reaches the user.
        An overflowing initial scene remains paused with an actionable error.
        """
        error = None
        settings = self.simulation.applied_settings
        try:
            self.simulation.launch_wave(settings)
            self.simulation.step(settings)
        except SpatialGridOverflowError as exc:
            error = exc
        finally:
            self.simulation.reset(settings)
            ti.sync()
        if error is not None:
            self.simulation.paused = True
            self.controls.last_error = error

    def run(self) -> Controls:
        """Process both input sources before physics and render even when paused.

        Frame durations include rendering and presentation. Physics warm-up and
        the first rendered frame (including shader compilation) are excluded
        from both the playback clock and the panel's steady-state FPS samples.
        """
        self.warm_up()
        frame_time = 0.0
        first_frame = True
        while self.window.running:
            started = monotonic()
            try:
                if not self.controls.update(self.window):
                    break
            except (ValueError, SpatialGridOverflowError):
                # Shared dispatch records recoverable keyboard errors.
                pass
            if self.controls.quit_requested:
                break

            self.panel.draw(self.gui, frame_time)
            if self.controls.quit_requested:
                break
            if not self.simulation.paused:
                try:
                    self.simulation.advance(frame_time, self.controls.settings)
                except SpatialGridOverflowError as exc:
                    self.simulation.paused = True
                    self.controls.last_error = exc

            self.controls.camera.apply(self.camera)
            self.scene.set_camera(self.camera)
            particles = self.simulation.particles
            self.renderer.render(
                self.scene, particle_positions=particles.positions,
                particle_radius=particles.collision_radius,
                particle_count=particles.active_count, particle_colors=particles.colors,
            )
            self.canvas.set_background_color(BACKGROUND_COLOR)
            self.canvas.scene(self.scene)
            self.window.show()
            elapsed = monotonic() - started
            frame_time = 0.0 if first_frame else elapsed
            first_frame = False
        return self.controls


def run() -> None:
    """Initialize Vulkan once and release the window on every exit path."""
    from .bootstrap import initialize_gpu

    initialize_gpu()
    window = create_window()
    try:
        Application(window).run()
    finally:
        window.destroy()
