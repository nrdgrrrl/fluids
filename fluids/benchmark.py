"""Opt-in spatial-grid and scalar-readback benchmark; never imported by the application.

Stages synchronize every measured pass and evolve one adaptive step per sample.
Native mode times the complete, unchanged Application.run loop, without extra
GPU synchronization. Run each layout in a fresh process; compare several runs
in alternating order. CPU results do not establish target-GPU performance.
"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from itertools import product
import json
import platform
from statistics import mean, median
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import taichi as ti

from .bootstrap import initialize_gpu
from .config import SMOOTHING_LENGTH_FACTOR, Settings
from .grid import CELL_SIZE
from .simulation import Simulation


@contextmanager
def _count_scalar_reads(sim):
    """Count actual host field reads without adding any GPU synchronization."""
    fields = {"overflow": sim.grid.overflow_count, "timestep": sim.dynamics._step_limit}
    reads = dict.fromkeys(fields, 0)
    field_type = type(fields["timestep"])
    original = field_type.__getitem__

    def read(field, key):
        for name, target in fields.items():
            if field is target:
                reads[name] += 1
        return original(field, key)

    with patch.object(field_type, "__getitem__", new=read):
        yield reads


@contextmanager
def _timestep_read_mode(sim, mode):
    """Duplicate control reproduces the old property read on every access."""
    if mode == "cached":
        yield
        return
    if mode != "duplicate":
        raise ValueError("timestep read mode must be cached or duplicate")
    solver_type = type(sim.dynamics)
    original = solver_type.stable_time_step.fget

    def uncached(solver):
        solver._cached_step_limit = None
        return original(solver)

    with patch.object(solver_type, "stable_time_step", new=property(uncached)):
        yield


def _read_report(reads, substeps):
    return {"counts": dict(reads), "total": sum(reads.values()),
            "per_completed_substep": {name: count / substeps if substeps else None
                                      for name, count in reads.items()}}


def candidate_checks(counts, span=1):
    """Checks before distance rejection, including self, for one neighbor pass.

    Use counts downloaded from a successful build, outside measured intervals.
    Each offset is counted once, with the same edge skipping as SpatialGrid.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = 0
    for offset in product(range(-span, span + 1), repeat=3):
        if any(abs(delta) >= size for delta, size in zip(offset, counts.shape)):
            continue
        source = tuple(slice(max(0, -delta), min(size, size - delta))
                       for delta, size in zip(offset, counts.shape))
        target = tuple(slice(max(0, delta), min(size, size + delta))
                       for delta, size in zip(offset, counts.shape))
        total += int(np.sum(counts[source] * counts[target]))
    return total


def _summary(samples):
    return {"samples": len(samples), "mean_ms": 1000 * mean(samples),
            "median_ms": 1000 * median(samples),
            "p95_ms": 1000 * float(np.percentile(samples, 95))}


def _candidates(sim):
    overflow = sim.grid.build()
    if overflow:
        raise RuntimeError(f"Cannot benchmark incomplete grid: {overflow} omitted insertions")
    checks = candidate_checks(sim.grid.cell_counts.to_numpy(), int(sim.grid._span[None]))
    return {"checks_per_pass": checks, "checks_per_particle": checks / sim.particles.active_count}


def measure_stages(sim, *, steps=40, warmup_steps=8, presteps=0, wave=False):
    """Synchronized stage wall times, not isolated GPU timestamps or frames."""
    sim.launch_wave()
    sim.step()  # Compile all passes, including the wave, then restore the scene.
    ti.sync()
    sim.reset()
    initial = _candidates(sim)
    if wave:
        sim.launch_wave()
    for _ in range(presteps + warmup_steps):
        sim.step()
    ti.sync()
    sampled = _candidates(sim)
    solver = sim.dynamics
    methods = ((sim.grid, "build", "grid"), (solver.density, "_compute", "density"),
               (solver, "_compute_forces", "forces"),
               (solver, "_read_stable_time_step", "timestep_readback"),
               (solver, "_integrate", "integration"),
               (solver.collisions, "resolve", "collisions"))
    timings = {label: [] for _, _, label in methods}

    def timed(label, method):
        def wrapper(*args, **kwargs):
            ti.sync()
            started = perf_counter()
            result = method(*args, **kwargs)
            ti.sync()
            timings[label].append(perf_counter() - started)
            return result
        return wrapper

    simulated_start = sim.simulated_time
    total = []
    with ExitStack() as stack:
        reads = stack.enter_context(_count_scalar_reads(sim))
        for owner, name, label in methods:
            stack.enter_context(patch.object(owner, name, new=timed(label, getattr(owner, name))))
        for _ in range(steps):
            started = perf_counter()
            sim.step()
            ti.sync()
            total.append(perf_counter() - started)
    return {"kind": "synchronized_evolving_stages", "initial_candidates": initial,
            "sample_start_candidates": sampled, "presteps": presteps, "warmup_steps": warmup_steps,
            "wave": wave, "stages": {name: _summary(values) for name, values in timings.items()},
            "diagnostic_substep": _summary(total),
            "simulated_seconds": sim.simulated_time - simulated_start,
            "scalar_reads": _read_report(reads, steps),
            "native_frames": "unmeasured", "driver_memory": "unmeasured"}


def measure_native(sim, *, warmup_seconds=8, sample_seconds=30, scenario="idle"):
    """Instrument Application's existing clock and advance call, preserving its loop."""
    from . import app

    window = app.create_window()
    frames, physics = [], []
    started = warmup_until = sample_start = None
    simulated_start = discarded_start = accumulated_start = substeps_start = None
    last_physics = 0.0
    stop = False
    substeps = 0

    class WindowProxy:
        @property
        def running(self):
            return not stop and window.running

        def __getattr__(self, name):
            return getattr(window, name)

    def frame_clock():
        nonlocal started, warmup_until, sample_start, stop, last_physics
        nonlocal simulated_start, discarded_start, accumulated_start, substeps_start
        now = perf_counter()
        if started is None:
            started = now
            last_physics = 0.0
            if warmup_until is not None and now >= warmup_until and sample_start is None:
                sample_start = now
                simulated_start, discarded_start = sim.simulated_time, sim.discarded_time
                accumulated_start, substeps_start = sim.accumulated_time, substeps
                for name in reads:
                    reads[name] = 0
        else:
            if warmup_until is None:
                warmup_until = now + warmup_seconds  # Exclude the first presented frame.
            elif sample_start is not None:
                frames.append(now - started)
                physics.append(last_physics)
                if now - sample_start >= sample_seconds:
                    stop = True
            started = None
        return now

    original_advance = sim.advance

    def advance(*args, **kwargs):
        nonlocal last_physics, substeps
        begin = perf_counter()
        result = original_advance(*args, **kwargs)
        last_physics = perf_counter() - begin
        substeps += result
        return result

    try:
        application = app.Application(WindowProxy(), simulation=sim)
        original_warm_up = application.warm_up

        def warm_up():
            original_warm_up()
            if scenario == "wave":
                sim.launch_wave()
            sim.paused = scenario == "paused"

        with (patch.object(application, "warm_up", new=warm_up),
              patch.object(app, "monotonic", new=frame_clock),
              patch.object(sim, "advance", new=advance),
              _count_scalar_reads(sim) as reads):
            application.run()
        if not frames:
            raise RuntimeError("Native run ended before collecting warmed frames")
        if application.controls.settings != sim.applied_settings or sim.last_error:
            raise RuntimeError("Settings changed or physics failed during the native benchmark")
        duration = sum(frames)
        return {"kind": "complete_native_frames", "scenario": scenario,
                "window_size": [1440, 900], "vsync": True, "warmup_seconds": warmup_seconds,
                "sample_complete": stop, "sample_seconds": duration, "fps": len(frames) / duration,
                "frame": _summary(frames), "physics_advance_wall": _summary(physics),
                "simulated_seconds": sim.simulated_time - simulated_start,
                "simulated_seconds_per_wall_second": (sim.simulated_time - simulated_start) / duration,
                "discarded_seconds": sim.discarded_time - discarded_start,
                "discarded_seconds_per_wall_second": (sim.discarded_time - discarded_start) / duration,
                "accumulated_seconds_start": accumulated_start,
                "accumulated_seconds_end": sim.accumulated_time,
                "substeps": substeps - substeps_start,
                "scalar_reads": _read_report(reads, substeps - substeps_start),
                "driver_memory": "unmeasured"}
    finally:
        window.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stages", "native"), default="stages")
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--layout", choices=("legacy", "fixed", "quality"), default="fixed")
    parser.add_argument("--timestep-reads", choices=("cached", "duplicate"), default="cached",
                        help="duplicate reproduces the previous uncached timestep readbacks")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--depth", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--presteps", type=int, default=0)
    parser.add_argument("--wave", action="store_true", help="launch a wave for stage measurements")
    parser.add_argument("--scenario", choices=("idle", "wave", "paused"), default="idle")
    parser.add_argument("--warmup-seconds", type=float, default=8)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--device-label", default="unrecorded", help="owner-supplied GPU/CPU model")
    parser.add_argument("--driver", default="unrecorded", help="owner-supplied driver version")
    parser.add_argument("--refresh-hz", type=float, help="owner-supplied desktop refresh rate")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.warmup_steps < 0 or args.presteps < 0:
        parser.error("steps must be positive; warmup-steps and presteps must be nonnegative")
    if (
        not np.isfinite(args.warmup_seconds) or args.warmup_seconds < 0
        or not np.isfinite(args.sample_seconds) or args.sample_seconds <= 0
    ):
        parser.error("warmup-seconds must be finite and nonnegative; sample-seconds must be finite and positive")
    if args.mode == "native" and args.backend != "vulkan":
        parser.error("native frames require Vulkan without CPU fallback")
    try:
        settings = Settings(quality=args.quality, water_depth=args.depth)
        cell_size = {"legacy": 1.6, "fixed": CELL_SIZE,
                     "quality": settings.particle_spacing * SMOOTHING_LENGTH_FACTOR}[args.layout]
        if args.backend == "vulkan":
            initialize_gpu()
        else:
            ti.init(arch=ti.cpu, enable_fallback=False, offline_cache=True)
            if ti.cfg.arch != ti.cpu:
                raise RuntimeError("CPU benchmark selected a different backend; unset TI_ARCH")
        sim = Simulation(settings, grid_cell_size=cell_size)
        report = {"backend": str(ti.cfg.arch), "os": platform.platform(),
                  "python": platform.python_version(), "taichi": ti.__version__,
                  "device_label": args.device_label, "driver": args.driver, "refresh_hz": args.refresh_hz,
                  "layout": args.layout, "cell_size_m": cell_size, "grid_shape": sim.grid.shape,
                  "timestep_reads": args.timestep_reads,
                  "logical_grid_bytes": sim.grid.storage_bytes, "particles": sim.particles.active_count,
                  "physics_settings": asdict(settings)}
        with _timestep_read_mode(sim, args.timestep_reads):
            if args.mode == "native":
                result = measure_native(sim, warmup_seconds=args.warmup_seconds,
                                        sample_seconds=args.sample_seconds, scenario=args.scenario)
            else:
                result = measure_stages(sim, steps=args.steps, warmup_steps=args.warmup_steps,
                                        presteps=args.presteps, wave=args.wave)
        report.update(result)
        print(json.dumps(report, indent=2))
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        print(f"fluids.benchmark: FAIL: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Details: {exc.__cause__}", file=sys.stderr)
        return 1
    finally:
        ti.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
