"""Opt-in native comparison: python -m fluids.render_benchmark --help.

Run stock and device cases in separate processes on the same physical Vulkan
GPU. The stock control reproduces 1.7.4 SceneV2 host staging. Instance timing
wrappers leave Application.run's physics, panel, camera and presentation order
intact. No forced sync is added to sampled frames. This is wall-clock timing,
not isolated GPU timestamps, and it requires a real desktop.
"""

import argparse
import json
import platform
import sys
from math import isfinite
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import taichi as ti
from taichi._lib import core as ti_core

from .app import Application, BACKGROUND_COLOR, create_window
from .bootstrap import initialize_gpu
from .config import DEFAULT_SETTINGS
from .rendering import (
    OUTLINE_COLOR, OUTLINE_VERTEX_COUNT, STATIC_VERTEX_CAPACITY, VBO_STRIDE,
    WATER_COLOR, Renderer, _container_outline, _static_mesh,
)
from .simulation import Simulation


class StockRenderer(Renderer):
    """Benchmark-only reproduction of the original field/SceneV2 draw path."""

    def __init__(self, world):
        # Allocate exactly the original renderer's fields, without device VBOs.
        self.vertices = ti.Vector.field(3, ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.colors = ti.Vector.field(3, ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.normals = ti.Vector.field(3, ti.f32, shape=STATIC_VERTEX_CAPACITY)
        self.outline_vertices = ti.Vector.field(3, ti.f32, shape=OUTLINE_VERTEX_COUNT)
        self.world = None
        self.vertex_count = 0
        self._last_particle_count = self._source_capacity = 0
        self.update_world(world)

    def update_world(self, world):
        if world == self.world:
            return False
        vertices, colors, normals, count = _static_mesh(world)
        self.vertices.from_numpy(vertices)
        self.colors.from_numpy(colors)
        self.normals.from_numpy(normals)
        if self.world is None or (world.domain_min != self.world.domain_min
                                  or world.domain_max != self.world.domain_max):
            self.outline_vertices.from_numpy(_container_outline(world))
        self.world, self.vertex_count = world, count
        return True

    def _draw_static(self, scene):
        scene.mesh(self.vertices, normals=self.normals, per_vertex_color=self.colors,
                   two_sided=True, vertex_count=self.vertex_count)
        scene.lines(self.outline_vertices, width=2.0, color=OUTLINE_COLOR)

    def _draw_particles(self, scene, positions, colors, radius, count):
        self._last_particle_count, self._source_capacity = count, positions.shape[0]
        scene.particles(positions, radius=radius, color=WATER_COLOR,
                        per_vertex_color=colors, index_count=count)

    def staging_info(self):
        return {
            "source": "HostMappedPtr (stock SceneV2 helper)",
            "backend": str(ti.cfg.arch),
            "particle_capacity": self._source_capacity,
            # Derived buffer sizes, not measured process or GPU memory.
            "particle_host_vbo_bytes_per_draw": self._source_capacity * VBO_STRIDE * 4,
            "static_host_vbo_bytes_per_frame": (STATIC_VERTEX_CAPACITY + OUTLINE_VERTEX_COUNT) * VBO_STRIDE * 4,
            "particles_packed_last_frame": self._source_capacity if self._last_particle_count else 0,
        }


def _rss_bytes():
    """Current Linux process RSS, or None on other hosts; never GPU memory."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _camera_state(app):
    camera = app.controls.camera
    return (camera.azimuth, camera.elevation, camera.distance, camera.target)


class FrameSamples:
    """Exclude compilation/first presentation and warm-up, then whole frames."""

    def __init__(self, warmup, seconds):
        self.warmup, self.seconds = warmup, seconds
        self.warmup_until = self.sample_started = None
        self.sample_finished = None
        self.started = 0.0
        self.physics = self.submission = 0.0
        self.frames = []
        self.memory = {}
        self.simulation_start = None

    def begin(self, now):
        self.started = now
        self.physics = self.submission = 0.0

    def finish(self, now, sim):
        if self.warmup_until is None:
            self.warmup_until = now + self.warmup
            return False
        if self.started < self.warmup_until:
            return False
        if self.sample_started is None:
            self.sample_started = self.started
        self.frames.append((now - self.started, self.physics, self.submission, sim.last_substeps))
        self.sample_finished = now
        return now - self.sample_started >= self.seconds

    def report(self):
        if not self.frames:
            raise RuntimeError("Window closed before any warmed samples were collected")
        count = len(self.frames)
        elapsed = self.sample_finished - self.sample_started
        return {
            "frames": count, "sample_seconds": elapsed, "fps": count / elapsed,
            "mean_frame_ms": 1000 * elapsed / count,
            "mean_physics_advance_ms": 1000 * sum(row[1] for row in self.frames) / count,
            "mean_render_submission_ms": 1000 * sum(row[2] for row in self.frames) / count,
            "mean_substeps": sum(row[3] for row in self.frames) / count,
        }


def _verify_device_submission(renderer, active_count):
    """Read metadata only; assert the exact sources the core receives."""
    for info in (renderer._mesh_info, renderer._outline_info, renderer._particle_info):
        if info is None or not info.valid or info.field_source != ti_core.FieldSource.TaichiNDarray:
            raise RuntimeError("Draw data is not a device ndarray")
    info = renderer._particle_info
    if info.shape != [active_count, VBO_STRIDE] or info.num_elements != active_count * VBO_STRIDE:
        raise RuntimeError("Particle draw exposes data beyond the active prefix")


def _reset_checks(app, settings, count):
    """Measure ten water/city resets after sampling, with rendered frames."""
    renderer, sim = app.renderer, app.simulation
    buffers = None if isinstance(renderer, StockRenderer) else (
        renderer._mesh_vbo, renderer._outline_vbo, renderer._particle_vbo,
    )
    camera, paused = _camera_state(app), sim.paused
    results = []
    sim.paused = True
    try:
        # No new fields or ndarrays are allowed during these resets/draws.
        with (
            patch.object(ti, "field", side_effect=RuntimeError("Reset allocated a field")),
            patch.object(ti.Vector, "field", side_effect=RuntimeError("Reset allocated a vector field")),
            patch.object(ti.Vector, "ndarray", side_effect=RuntimeError("Reset allocated a device VBO")),
        ):
            for index in range(count):
                if not app.window.running:
                    raise RuntimeError("Window closed before reset validation finished")
                requested = settings.with_updates(city_seed=index + 1, city_density=index % 2,
                                                  water_depth=1.0 if index % 2 else 2.0)
                started = perf_counter()
                sim.reset(requested)
                renderer.update_world(sim.world)
                ti.sync()
                reset_ms = 1000 * (perf_counter() - started)
                particles = sim.particles
                app.controls.camera.apply(app.camera)
                app.scene.set_camera(app.camera)
                renderer.render(app.scene, particle_positions=particles.positions,
                                particle_colors=particles.colors, particle_count=particles.active_count,
                                particle_radius=particles.collision_radius)
                app.canvas.set_background_color(BACKGROUND_COLOR)
                app.canvas.scene(app.scene)
                app.window.show()
                ti.sync()
                if buffers is not None:
                    _verify_device_submission(renderer, particles.active_count)
                    if buffers != (renderer._mesh_vbo, renderer._outline_vbo, renderer._particle_vbo):
                        raise RuntimeError("Reset replaced a device staging buffer")
                if _camera_state(app) != camera or not sim.paused:
                    raise RuntimeError("Reset changed camera or pause selection")
                results.append({"reset": index + 1, "particles": particles.active_count,
                                "reset_upload_ms": reset_ms,
                                "reset_and_present_ms": 1000 * (perf_counter() - started),
                                "process_rss_bytes": _rss_bytes(), "staging": renderer.staging_info()})
    finally:
        sim.paused = paused
    return results


def run_benchmark(window, *, staging="device", scenario="paused", quality="medium",
                  warmup=8.0, seconds=30.0, resets=10, capture=None):
    """Use an initialized Vulkan backend and the application's normal loop."""
    if ti.cfg.arch != ti.vulkan:
        raise RuntimeError("The rendering benchmark requires Vulkan; CPU timing is not supported")
    settings = DEFAULT_SETTINGS.with_updates(quality=quality)
    sim = Simulation(settings)
    renderer = (Renderer if staging == "device" else StockRenderer)(sim.world)
    app = Application(window, simulation=sim, renderer=renderer)
    sim.paused = scenario == "paused"
    samples = FrameSamples(warmup, seconds)
    camera = _camera_state(app)
    original_update, original_render = app.controls.update, renderer.render
    original_advance, original_show, original_warmup = sim.advance, window.show, app.warm_up

    def update(*args, **kwargs):
        samples.begin(perf_counter())
        # Sample RSS/timing state before the first measured frame's work.
        if (samples.warmup_until is not None and samples.started >= samples.warmup_until
                and samples.sample_started is None):
            samples.memory["sample_start_rss_bytes"] = _rss_bytes()
            samples.simulation_start = (sim.simulated_time, sim.discarded_time)
        return original_update(*args, **kwargs)

    def render(*args, **kwargs):
        started = perf_counter()
        result = original_render(*args, **kwargs)
        samples.submission = perf_counter() - started
        return result

    def advance(*args, **kwargs):
        started = perf_counter()
        result = original_advance(*args, **kwargs)
        samples.physics = perf_counter() - started
        return result

    def show():
        if samples.warmup_until is None and capture is not None:
            # Optional pixel readback belongs to the excluded first frame.
            window.save_image(str(capture))
        original_show()
        done = samples.finish(perf_counter(), sim)
        if sim.last_error is not None:
            raise RuntimeError(f"Benchmark physics failed: {sim.last_error}")
        if (sim.applied_settings != settings or app.controls.settings != settings
                or sim.paused != (scenario == "paused") or _camera_state(app) != camera):
            raise RuntimeError("Benchmark settings, camera or pause selection changed during sampling")
        if staging == "device":
            _verify_device_submission(renderer, sim.particles.active_count)
        if done:
            app.controls.quit_requested = True

    def warm_up():
        original_warmup()
        if scenario == "wave":
            sim.launch_wave()

    with (
        patch.object(app.controls, "update", new=update),
        patch.object(renderer, "render", new=render),
        patch.object(sim, "advance", new=advance),
        patch.object(window, "show", new=show),
        patch.object(app, "warm_up", new=warm_up),
    ):
        app.run()
    if not samples.frames or samples.sample_finished - samples.sample_started < seconds:
        raise RuntimeError("Window closed before the requested sample duration completed")
    samples.memory["sample_end_rss_bytes"] = _rss_bytes()
    report = {
        "staging": staging, "scenario": scenario, "quality": quality,
        "particles": sim.particles.active_count, "particle_capacity": sim.particles.capacity,
        "window_size": list(window.get_window_shape()), "vsync": True,
        "warmup_seconds": warmup, "requested_sample_seconds": seconds,
        "first_frame_capture": None if capture is None else str(capture),
        "timing": samples.report(), "process_memory": samples.memory,
        "gpu_memory": "unmeasured; collect driver counters on the target desktop",
        "simulation_start": {"simulated_seconds": samples.simulation_start[0],
                             "discarded_seconds": samples.simulation_start[1]},
        "simulation_end": {"simulated_seconds": sim.simulated_time, "discarded_seconds": sim.discarded_time},
        "staging_metadata": renderer.staging_info(),
    }
    report["water_city_resets"] = _reset_checks(app, settings, resets)
    return report


def _positive_seconds(value):
    seconds = float(value)
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", choices=("stock", "device"), default="device")
    parser.add_argument("--scenario", choices=("paused", "active", "wave"), default="paused")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--warmup", type=_positive_seconds, default=8.0)
    parser.add_argument("--seconds", type=_positive_seconds, default=30.0)
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--gpu", default="unrecorded", help="physical GPU model (owner supplied)")
    parser.add_argument("--driver", default="unrecorded", help="driver version (owner supplied)")
    parser.add_argument("--refresh-hz", type=_positive_seconds, help="desktop refresh rate (owner supplied)")
    parser.add_argument("--output", type=Path, help="write JSON report in addition to stdout")
    parser.add_argument("--capture", type=Path, help="save first scene image before warm-up (e.g. device.png)")
    args = parser.parse_args(argv)
    if args.resets < 0:
        parser.error("--resets must be nonnegative")
    window = None
    try:
        initialize_gpu()
        window = create_window()
        report = run_benchmark(window, staging=args.staging, scenario=args.scenario, quality=args.quality,
                               warmup=args.warmup, seconds=args.seconds, resets=args.resets, capture=args.capture)
        report.update(os=platform.platform(), python=platform.python_version(), taichi=ti.__version__,
                      gpu=args.gpu, driver=args.driver, refresh_hz=args.refresh_hz)
        output = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.write_text(output + "\n")
        print(output)
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        print(f"fluids.render_benchmark: FAIL: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Details: {exc.__cause__}", file=sys.stderr)
        return 1
    finally:
        if window is not None:
            window.destroy()
        ti.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
