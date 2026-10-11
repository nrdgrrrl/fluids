# fluids

Native Python desktop application for an interactive 3D coastal-wave sandbox.
This first checkpoint opens an empty Taichi window; particles, buildings, and
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

Dependencies are pinned to the issue's baseline: Taichi 1.7.4 and NumPy 1.26.4.
Installation success does not establish GPU or driver compatibility. Verify the
application on each target desktop; no simulation or rendering validation is
claimed before testing on actual target hardware.

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

These checks mock GPU and window operations; they verify startup policy, failure
reporting, and loop shutdown without claiming hardware rendering coverage. On
the target desktop, run `fluids`, resize the window, close it, and confirm the
process exits. With `TI_ENABLE_VULKAN=0`, running `fluids` must report the Vulkan
error and exit without opening a window or falling back to CPU.
