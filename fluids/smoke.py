"""Opt-in Vulkan invariants: ``python -m fluids.smoke`` (no window).

This diagnostic deliberately downloads fields for inspection. It is never
imported by the application and is not an FPS benchmark or visual validation.
The CLI uses the production backend policy and never falls back to CPU.
"""

import argparse
import json
import platform
import sys
from unittest.mock import patch

import numpy as np
import taichi as ti

from .bootstrap import initialize_gpu
from .config import DEFAULT_SETTINGS, MAX_SPEED, REST_DENSITY
from .grid import CELL_CAPACITY
from .rendering import Renderer
from .simulation import Simulation, SpatialGridOverflowError


TOLERANCE = 2e-5  # float32 projection roundoff, in metres


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _state_fields(sim):
    state = sim.particles
    return (state.positions, state.velocities, state.accelerations,
            state.densities, state.pressures, state.colors)


def _check_state(sim, expected_count, expected_mass, *, density_ready=True):
    state = sim.particles
    _require(state.active_count == expected_count == int(state.count[None]),
             "Active particle count changed")
    _require(np.isclose(state.mass * expected_count, expected_mass, rtol=1e-12),
             "Total water mass changed")
    arrays = [field.to_numpy()[:expected_count] for field in _state_fields(sim)]
    _require(all(np.isfinite(array).all() for array in arrays), "Nonfinite fluid state")
    positions, velocities, _, densities, pressures, colors = arrays
    _require((densities > 0).all() if density_ready else (densities >= 0).all(),
             "Invalid density")
    _require((pressures >= 0).all(), "Negative pressure")
    _require((np.linalg.norm(velocities, axis=1) <= MAX_SPEED + TOLERANCE).all(),
             "Velocity safeguard exceeded")
    _require(((colors >= 0) & (colors <= 1)).all(), "Invalid particle colors")
    radius = state.collision_radius
    lower = np.asarray(sim.world.domain_min) + radius
    upper = np.asarray(sim.world.domain_max) - radius
    _require(((positions >= lower - TOLERANCE) & (positions <= upper + TOLERANCE)).all(),
             "Particle escaped container")
    for box in sim.world.boxes:
        inside = np.all((positions > np.asarray(box.minimum) - radius + TOLERANCE)
                        & (positions < np.asarray(box.maximum) + radius - TOLERANCE), axis=1)
        _require(not inside.any(), "Particle penetrated land or building")


def _check_collisions(sim, count, mass):
    """Force contact with all container faces, the land top, and a building."""
    state, world = sim.particles, sim.world
    radius = state.collision_radius
    lower, upper = np.array(world.domain_min) + radius, np.array(world.domain_max) - radius
    probes = []
    for axis in range(3):
        for side, bounds in enumerate((lower, upper)):
            direction = 1 if side == 0 else -1
            point = np.array([-10.0, 5.0, 10.0])
            point[axis] = bounds[axis] - direction * 0.1
            velocity = np.zeros(3)
            velocity[axis] = -2 * direction
            projected = point.copy()
            projected[axis] = bounds[axis]
            probes.append((point, velocity, projected, axis, direction))
    # The land probe sits in the border street, clear of all buildings.
    point = np.array([0.05, world.land.maximum[1] + radius - 0.01, 0.3])
    projected = point.copy()
    projected[1] = world.land.maximum[1] + radius
    probes.append((point, [0, -2, 0], projected, 1, 1))
    _require(bool(world.buildings), "Collision smoke scene needs a building")
    box = world.buildings[0]
    point = (np.array(box.minimum) + box.maximum) / 2
    point[0] = box.minimum[0] - radius + 0.01
    projected = point.copy()
    projected[0] = box.minimum[0] - radius
    probes.append((point, [2, 0, 0], projected, 0, -1))
    for index, (point, velocity, _, _, _) in enumerate(probes):
        state.positions[index], state.velocities[index] = point, velocity
    sim.dynamics.collisions.resolve()
    for index, (_, _, expected, axis, direction) in enumerate(probes):
        _require(np.allclose(state.positions[index], expected, rtol=0, atol=TOLERANCE),
                 f"Collision projection failed for probe {index}")
        speed = float(state.velocities[index][axis])
        _require(np.isclose(speed, 2 * direction * sim.applied_settings.collision_restitution,
                            rtol=0, atol=TOLERANCE), f"Collision bounce failed for probe {index}")
    _check_state(sim, count, mass, density_ready=False)


def run_checks(*, steps=120):
    """Run checks on an already initialized backend; CLI always requires Vulkan."""
    _require(isinstance(steps, int) and not isinstance(steps, bool) and steps > 0,
             "steps must be a positive integer")
    sim = Simulation(DEFAULT_SETTINGS)
    renderer = Renderer(sim.world)
    state = sim.particles
    count, mass = state.active_count, state.mass * state.active_count
    _require(np.isclose(mass, REST_DENSITY * state.lattice.volume), "Incorrect initial water mass")

    # Warm compilation, restore the initial scene, and exercise a real impulse.
    sim.launch_wave()
    sim.step()
    sim.reset()
    sim.launch_wave()
    velocities = state.velocities.to_numpy()[:count]
    _require((velocities[:, 0] > 0).any() and (velocities[:, 1] > 0).any(),
             "Default wave did not add forward and lift velocity")
    _check_state(sim, count, mass, density_ready=False)
    for step in range(steps):
        sim.step()
        if (step + 1) % 10 == 0 or step + 1 == steps:
            _check_state(sim, count, mass)
    elapsed = sim.simulated_time
    print(f"PASS: wave and finite state ({count:,} particles, {steps} steps)", flush=True)

    sim.reset()
    _check_collisions(sim, count, mass)
    print("PASS: container, land, and building collisions", flush=True)

    # Materialize fields before guarding allocation. Reset both water and city
    # ten times on the same objects, including renderer geometry uploads.
    fields = _state_fields(sim) + (
        state.count, sim.grid.cell_counts, sim.grid.particle_indices, sim.grid.overflow_count,
        sim.grid._origin, sim.grid._radius, sim.grid._span, sim.grid._ready, sim.dynamics._step_limit,
        sim.dynamics.collisions.building_minimum, sim.dynamics.collisions.building_maximum,
        sim.dynamics.collisions.building_occupied,
        renderer.vertices, renderer.colors, renderer.normals, renderer.outline_vertices,
    )
    original_shapes = [field.shape for field in fields]
    with (
        patch.object(ti, "field", side_effect=RuntimeError("Reset allocated a scalar field")),
        patch.object(ti.Vector, "field", side_effect=RuntimeError("Reset allocated a vector field")),
    ):
        sim.paused = True
        for seed in range(1, 11):
            settings = DEFAULT_SETTINGS.with_updates(city_seed=seed, city_density=0 if seed % 2 else 1)
            sim.reset(settings)
            renderer.update_world(sim.world)
            _require(sim.paused and sim.world is renderer.world, "Reset lost pause state or geometry")
            _check_state(sim, count, mass, density_ready=False)
            sim.step()  # Explicit stepping still works while paused.
            _check_state(sim, count, mass)
    _require([field.shape for field in fields] == original_shapes, "Field capacities changed")
    print("PASS: ten water/city resets with no new simulation or renderer fields", flush=True)

    # Collapse the active prefix into one bin without changing count or mass.
    # A failed step must preserve every particle field and simulated time.
    sim.reset(DEFAULT_SETTINGS)
    state.positions.fill(0)
    before = [field.to_numpy() for field in _state_fields(sim)]
    try:
        sim.step()
    except SpatialGridOverflowError as exc:
        _require(exc.overflow_count == count - CELL_CAPACITY, "Incorrect grid overflow count")
    else:
        raise RuntimeError("Overfull bin did not stop physics")
    _require(sim.simulated_time == 0 and sim.last_substeps == 0, "Overflow advanced physics time")
    for field, expected in zip(_state_fields(sim), before):
        _require(np.array_equal(field.to_numpy(), expected), "Overflow mutated particle state")
    sim.reset()
    sim.step()
    _require(sim.last_error is None and sim.grid.overflow_count[None] == 0, "Reset did not recover overflow")
    _check_state(sim, count, mass)
    ti.sync()
    print("PASS: exact grid-overflow detection, physics gate, and reset recovery", flush=True)
    return {"backend": str(ti.cfg.arch), "os": platform.platform(), "quality": DEFAULT_SETTINGS.quality,
            "particles": count, "total_mass_kg": mass, "steps": steps,
            "simulated_seconds": elapsed, "water_city_resets": 10}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120, help="positive number of physics steps (default: 120)")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    try:
        initialize_gpu()
        report = run_checks(steps=args.steps)
        print(json.dumps(report, indent=2))
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        print(f"fluids.smoke: FAIL: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Details: {exc.__cause__}", file=sys.stderr)
        return 1
    finally:
        ti.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
