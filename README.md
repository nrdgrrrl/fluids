# fluids

Native Python desktop application for an interactive 3D coastal-wave sandbox.
This first checkpoint opens an empty Vulkan window; particles, buildings, and
collisions will be added in later issues. It is not a validated engineering model.

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

The native 1440 × 900 window is resizable. Close it to exit. Vertical sync is
enabled. Importing `fluids` does not initialize the GPU or open a window.

Taichi 1.7.4 and NumPy 1.26.4 remain pinned to the issue's baseline. GLFW 2.10.0
creates the resizable native window; wgpu 0.31.1 presents through Vulkan and
updates the swapchain to the framebuffer size after each resize. Taichi's Vulkan
compute backend is still required. Minimized windows pause drawing until restored.
Installation success does not establish GPU or driver compatibility. Verify the
application on each target desktop; no simulation or rendering validation is
claimed before testing on actual target hardware.

## Startup failures

Vulkan is required. Fluids disables Taichi's CPU fallback and rejects conflicting
`TI_ARCH` values. Unset `TI_ARCH` or set it to `vulkan` if needed. The native renderer
also requires Vulkan; unset `WGPU_BACKEND_TYPE` or set it to `Vulkan`. Backend failures
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

By default, these checks mock GPU and window operations. They verify startup
policy, failure reporting, surface updates after resizing, minimize/restore, and
loop shutdown. To also exercise a real native window on a Vulkan desktop, run:

```sh
FLUIDS_TEST_NATIVE=1 python -m unittest discover -s tests -v
```

In PowerShell, set `$env:FLUIDS_TEST_NATIVE = "1"` before running the test command.
The native test shrinks and enlarges the window, verifies the resizable attribute
and Vulkan backend, and reads rendered pixels at opposite corners of each new
framebuffer. Headless Linux CI can use Xvfb with Mesa's software Vulkan driver;
`xvfb` and `mesa-vulkan-drivers` belong in the runner's installation-level runtime
packages. This checks native resizing and rendering without establishing physical
GPU compatibility. On the target desktop, also run `fluids`, drag the window borders,
minimize/restore it, close it, and confirm the process exits. With
`TI_ENABLE_VULKAN=0`, running `fluids` must report the Vulkan error and exit
without opening a window or falling back to CPU.
