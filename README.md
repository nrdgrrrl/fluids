# fluids

Native Python desktop application for an interactive 3D coastal-wave sandbox.
The application displays an ocean of SPH particles, a generated coastal city,
and a parameter panel. It is not a validated engineering model.

## Install and run

Use Python 3.11 on a desktop with a Vulkan-capable GPU and a compatible driver:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
fluids
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell. `python -m fluids` is also supported.

The native 1440 × 900 Taichi GGUI window uses Vulkan and vertical sync. Taichi
1.7.4's scene window has a fixed size. Close it or press Escape to exit.
Importing `fluids` does not initialize the GPU or open a window.

Use Space to pause/resume, N to step while paused, W to launch a wave toward the
coast, R to apply pending settings and reset, and C to advance the displayed
city seed and reset water. Reset preserves pause state and camera position. A
wave launched while paused changes velocity; particles move on step or resume.
Structural sliders marked `[reset]` remain pending until reset. Other controls
apply live. Left-drag orbits, right-drag pans, +/− zoom, and 0 restores the view.
The panel has matching action buttons; H shows all shortcuts. Camera and panel
input continue while paused. Grid overflow pauses physics until a successful
reset; its diagnostic appears in the panel.

Startup compiles the physics and wave kernels, then restores the initial scene.
This warm-up and the first rendered frame are excluded from playback timing and
steady-state FPS. Verify rendering and interaction on each target desktop;
installation success does not establish GPU or driver compatibility.

## Startup failures

Vulkan is required. Fluids disables Taichi's CPU fallback and rejects conflicting
`TI_ARCH` values. Unset `TI_ARCH` or set it to `vulkan` if needed. Backend failures
exit with a nonzero status and the message:

```text
Vulkan initialization failed. Install a compatible GPU driver and run this application in a desktop graphics session.
```

The underlying error is printed as additional diagnostic detail. Install or update
the GPU vendor's Vulkan driver and ensure the Vulkan runtime is available. A
headless session cannot display this application. On Linux, use an X11 desktop
or XWayland with `DISPLAY` set. Window creation failures also report an actionable
error; the application does not switch rendering engines or compute backends.

## Verification

Install into the Python 3.11 environment, then run:

```sh
python -m unittest discover -s tests -v
```

The checks exercise physics on CPU and mock window operations. They verify startup
policy, action ordering, reset and overflow recovery, frame timing, and shutdown.
To also exercise the legacy resizable clear-window backend on a Vulkan desktop, run:

```sh
FLUIDS_TEST_NATIVE=1 python -m unittest discover -s tests -v
```

In PowerShell, set `$env:FLUIDS_TEST_NATIVE = "1"` before running the test command.
The native test shrinks and enlarges the window, verifies the resizable attribute
and Vulkan backend, and reads rendered pixels at opposite corners of each new
framebuffer. Headless Linux CI can use Xvfb with Mesa's software Vulkan driver;
`xvfb` and `mesa-vulkan-drivers` belong in the runner's installation-level runtime
packages. This checks native resizing and rendering without establishing physical
GPU compatibility. On the target desktop, also run `fluids`, launch a wave, pause
and step, edit settings and reset, randomize the city, orbit/pan while paused,
close it, and confirm the process exits. With
`TI_ENABLE_VULKAN=0`, running `fluids` must report the Vulkan error and exit
without opening a window or falling back to CPU.
