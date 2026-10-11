# fluids

A native Python desktop sandbox for a 3D coastal wave, GPU SPH water particles,
and a seeded procedural city. This is an interactive approximation, not a
validated engineering or flood prediction model.

## Setup and launch

Use **Python 3.11**, a Vulkan-capable GPU with its vendor's Vulkan driver, and a
desktop graphics session. The package pins Taichi 1.7.4, NumPy 1.26.4, GLFW 2.10.0,
and wgpu 0.31.1. Linux requires X11 or XWayland with `DISPLAY` set. A headless
session can run the CPU tests but cannot display the application.

From the repository on Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
fluids
```

On Windows, use PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install .
fluids
```

`python -m fluids` launches the same application. Developers can use
`python -m pip install -e .` for an editable installation. The native window is
1440 × 900 with vertical sync; Taichi 1.7.4's GGUI scene window has a fixed size.
Importing `fluids` does not initialize a GPU or open a window.

Water starts offshore at rest. Press **W** (or **Launch wave**) to add the default
impulse toward the city. The shoreline is x = 0, the ocean is at negative x,
the city is at positive x, and y points up. Lengths are metres and time is seconds.

## Controls

| Input | Action |
| --- | --- |
| Space | Pause/resume physics |
| N | Advance one adaptive physics step while paused |
| W | Add one offshore wave impulse using current wave controls |
| R | Apply pending settings and reset water and city |
| C | Increment the displayed city seed and reset water and city |
| H | Show/hide shortcut help |
| Left drag | Orbit the scene |
| Right drag | Pan the camera target |
| + / − | Zoom in/out (`=` also zooms in) |
| 0 | Restore the initial camera |
| Escape / window close | Exit |

The panel provides matching physics, wave, reset, city, and help buttons.
Camera navigation and parameter editing work while paused. Drags starting over
the panel edit controls without moving the camera. Reset preserves the camera
and pause selection. Launching a wave while paused changes velocity; water moves
when stepped or resumed. Repeated launches add impulses, subject to the speed cap.
“Randomize city” advances the seed deterministically; the same seed and city
settings reproduce the same geometry.

## Settings and recovery

Use the **Water**, **City**, and **Physics** selectors in the panel. Sliders
labelled `[reset]` change the requested scene; `*` and “Pending reset” indicate
an unapplied edit. Press R or Apply / reset to rebuild it. C also applies pending
settings. Other controls act live on the next relevant operation.

| Group | Requires reset | Live settings |
| --- | --- | --- |
| Water | Water depth (0.5–4 m), particle quality (low/medium/high) | Wave forward speed, width, lift height; these affect the next launch |
| City | Seed, lot occupancy density, minimum/maximum building height | None |
| Physics | None | Gravity, viscosity, damping, collision bounce/friction, playback speed |

Legacy wave amplitude, period, and direction remain in the Water panel for
compatibility but **do not affect launched impulses**. The impulse always travels
toward +x from a band centred at x = −12. Lift height sets launch energy rather
than guaranteeing a measured crest height. Crossing the building height bounds
moves the other bound too, keeping the range valid. The uint32 seed is edited
as leading digits and the last four digits.

Playback speed scales requested simulation time. Zero stops integration; Space
controls pause independently, and N still steps while paused at zero speed.
The panel shows FPS, active particles, simulated time, substeps, discarded time,
pending edits, and errors.

If a spatial bin overflows, physics stops and the application pauses with an
error. Camera, help, and settings remain usable. Lower quality or water depth
and reset, then resume; pause/step cannot bypass this recovery requirement.
Rejected settings or particle capacity proposals leave the existing scene intact.

Vulkan is required, with CPU fallback disabled. Unset `TI_ARCH` or set it to
`vulkan`. Backend failures exit nonzero with diagnostic details and:

```text
Vulkan initialization failed. Install a compatible GPU driver and run this application in a desktop graphics session.
```

Window creation failures also exit nonzero with a desktop/driver diagnostic.
Check the GPU vendor driver, Vulkan runtime, and Linux `DISPLAY`/XWayland setup.
Installation alone does not establish GPU or display compatibility.

## Verification

Run the focused CPU contracts first, then the full suite:

```sh
python -m unittest discover -s tests -p 'test_core.py' -v
python -m unittest discover -s tests -v
```

The focused checks cover settings defaults, inclusive bounds, invalid edits,
reset policy, deterministic city/particle generation, street and mesh geometry,
water bounds, mass, and particle capacity. The full suite runs real physics
kernels on CPU by default and mocks desktop input/rendering. It also runs the
smoke diagnostic's logic on CPU; this does not count as a GPU smoke pass.

On a supported Vulkan machine, explicitly invoke the windowless GPU diagnostic:

```sh
python -m fluids.smoke
```

It uses the application's Vulkan initialization policy without CPU fallback.
It exercises the default medium scene and wave for 120 physics steps, checking
finite positions, velocities, accelerations, densities, pressures, and colors;
particle count and total mass; container/land/building collisions; ten water/city
resets without new simulation or renderer fields; forced bin overflow, unchanged
fluid state on failure, and reset recovery. `--steps N` changes the positive step
count. Invariant or backend failures exit nonzero; success prints PASS lines and
a JSON summary. Field downloads are confined to this diagnostic and tests.
This command checks compute invariants, not rendering, coast arrival, or FPS.

The existing optional `FLUIDS_TEST_NATIVE=1` test checks the legacy resizable
GLFW/wgpu clear-window backend. It does not verify the application's GGUI scene.
Software Vulkan/Xvfb results do not establish physical GPU performance.

## Manual pass and performance record

Run `fluids` on the target desktop after CPU tests and the GPU smoke check.
Record each item as pass/fail/unexecuted in the implementation PR:

1. At default medium quality and 1440 × 900, launch W and visually follow the
   wave to the coast and through streets around buildings. Record any overflow
   or failure to reach land.
2. Perform ten consecutive R resets and ten C city regenerations. Check water
   and mesh replacement, stable particle/field capacities and memory after
   warm-up, preserved camera/pause state, and absence of crashes. The diagnostic
   guards field allocation; observe driver/process memory on the desktop too.
3. Verify pause/resume, repeated N steps, launch while paused, camera orbit/pan,
   both zoom directions, restore view, panel drag capture, H help, and exit.
4. Edit every Water, City, and Physics control. Confirm live effects, pending
   reset markers, depth/quality rebuilds, height bounds, seeds, and zero/fast
   playback. Confirm legacy wave controls are identified as inactive.
5. Set high quality with 4 m depth and reset to exercise overflow recovery.
   Lower quality/depth, reset, and resume; confirm the error clears and input
   remains usable during recovery. For startup failure handling, launch with
   `TI_ENABLE_VULKAN=0` and confirm a nonzero exit without a window/CPU fallback.

Compilation and the first rendered frame are excluded from the application
playback clock and FPS samples. Wait for warm-up and stable rendering before
measuring. Start with the unchanged default medium scene (18,360 particles) at
1440 × 900, launch one default wave, and sample the panel for at least 30 seconds.
Record a separate idle sample if useful. Report FPS range/typical value, sample
duration, vsync/display refresh, and simulated/discarded time at the beginning
and end. The FPS display is smoothed; report it as such. Growing discarded time
means physics is dropping requested time, even if rendering appears smooth.

The PR must include **GPU model, OS, driver version, quality, window size,
particle count, warmed FPS, and whether discarded catch-up time grows**, plus
CPU/smoke/manual outcomes. If medium cannot remain interactive on the target
GPU, measure low quality and lower the default based on those results before
merging. GPU computation must remain on GPU. No universal FPS target is claimed.

### Issue #15 implementation environment

The implementation container runs Debian GNU/Linux 12, x86_64, Python 3.11.2.
It exposes no `/dev/dri` or NVIDIA GPU device, Vulkan ICD, or desktop `DISPLAY`.
The focused CPU checks passed (8 tests); the full suite passed (147 tests run,
1 desktop-only test skipped). Both `python -m fluids.smoke` and `fluids` were
attempted and exited with status 1: Vulkan instance creation is unsupported.
Neither fell back to CPU or opened a window.
The diagnostic was separately exercised on CPU for all 120 steps and ten city
resets; its invariants passed with 18,360 particles and 905,810 kg total mass.
That CPU result is not a target-GPU validation or performance measurement.
Target GPU model and driver, warmed rendering FPS, and discarded-time trend are
**unmeasured**. A supported target machine has not been validated here; the
visual pass and target-GPU smoke pass remain required before merging. Medium
remains the default because no target-GPU measurement supports a change.

## Approximation and resource limits

- Water uses weakly compressible SPH, an artificial 20 m/s sound speed,
  nonnegative capped pressure, density floors, and acceleration/velocity caps.
  Damping and collision friction dissipate energy; these safeguards do not
  establish accuracy or exact momentum/energy conservation.
- Buildings and the flat 0.25 m land slab are static axis-aligned boxes.
  Collisions project particle spheres out of expanded boxes. There is no
  bathymetry, erosion, building damage, drainage, open ocean boundary, or
  validated turbulence/air model. The container is closed on all six sides.
- Water renders as colored spheres rather than a reconstructed free surface.
  Lift/forward impulses and pale splash colors are visual approximations;
  neither crest height nor flood depth is a prediction.
- Particle capacity is fixed at 131,072. The largest supported initial lattice
  has 114,000 particles, but particle capacity and grid capacity are separate
  constraints. Each of the 30 × 10 × 20 spatial bins holds 256 indices. High
  quality at 4 m depth overflows bins even at initialization (up to 343
  particles in a cell); compression can overflow other scenes too.
- Physics uses adaptive timesteps between 10 µs and 1/120 s, accepts at most
  0.1 s of each frame and runs at most 16 substeps. Excess requested time is
  discarded rather than integrated in an unsafe large step. High quality,
  deep water, fast waves, high viscosity, or fast playback can reduce effective
  simulation speed. Real interaction and performance depend on GPU and driver.
