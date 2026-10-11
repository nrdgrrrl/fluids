# fluids

A native GPU coastal-wave sandbox: a desktop application that simulates
three-dimensional fluid particles flowing around procedural buildings.

It is an interactive visual simulation, not a validated engineering model.

## Status

This is the first checkpoint: packaging, Vulkan backend bootstrap, and a
resizable native window. The simulation and rendering are added by later
work, so the window currently shows an empty background.

## Requirements

- Python 3.11
- A GPU and driver with working Vulkan support, in a desktop graphics session

The dependency pins in `pyproject.toml` (`taichi==1.7.4`, `numpy==1.26.4`)
are a baseline to verify on your machine, not a claim of universal driver
compatibility.

## Install and run

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install .
fluids
```

`fluids` opens the application window; closing the window exits the process.
Dragging a window edge resizes it: Taichi's GGUI window recreates the Vulkan
swap chain from its framebuffer-size callback, so frames keep presenting at
the new size. That path needs a display, so confirm it on target hardware by
dragging an edge while `fluids` runs.

To check the GPU backend without opening a window:

```sh
fluids --check-gpu
```

If Vulkan is unavailable, both commands print an actionable error and exit
with status 1. The application never falls back to running fluid physics on
the CPU, because a silent fallback would hide an unsupported installation.

## Tests

```sh
pip install -e '.[dev]'
pytest
```

The tests cover packaging-independent behaviour — backend policy, error
handling, and the entry point — against a Taichi stub, so they run without a
GPU. GPU simulation, rendering, and window resizing can only be verified by
running the application on target hardware.
