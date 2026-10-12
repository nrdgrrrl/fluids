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

### Device vertex staging (issue #37)

The renderer owns reusable 12-float `ti.Vector.ndarray` VBOs. Static positions,
normals and colors are packed only on a changed world; the outline is packed
only on changed bounds. Particle packing visits only `particle_count`, and the
core receives both that prefix's shape and element count. For medium water,
the submitted particle data is **881,280 bytes** (18,360 × 48), rather than the
reserved **6,291,456 bytes** (131,072 × 48). These are calculated buffer sizes.
The device VBO retains solver capacity for allocation-free water/city resets;
a larger source capacity grows it after synchronizing outstanding uses.

Taichi 1.7.4's [SceneV2 helper](https://github.com/taichi-dev/taichi/blob/v1.7.4/python/taichi/ui/staging_buffer.py)
allocates a NumPy VBO on each draw. Passing an ndarray to the public scene
methods would still invoke that helper. Fluids instead passes `FieldInfo`
directly to the pinned core scene, with `TaichiNDarray` and the array's actual
`DeviceAllocation`. A local adapter supports both `get_device_allocation()`
and the installed core's `device_allocation()`, and fails explicitly on an
unsupported accessor or Taichi version; it changes no Taichi globals.

With Vulkan, Taichi's [core copy path](https://github.com/taichi-dev/taichi/blob/v1.7.4/taichi/ui/ggui/renderable.cpp)
enqueues a same-device buffer copy. It still copies static VBOs to its own
render buffers each frame. The core also reserves host-writable staging buffers;
the same-device path does not map or fill them with particle data. Thus the
claim is absence of particle-sized **host uploads during steady frames**, not
absence of every host-visible allocation in GGUI or zero-copy drawing.

```sh
python -m unittest discover -s tests -p 'test_rendering.py' -v
FLUIDS_TEST_PARTICLE_ARCH=vulkan python -m unittest discover -s tests -p 'test_rendering.py' -v
```

The CPU contracts use real packing kernels, `FieldInfo` and allocation objects,
with a mocked native scene. They verify active transfer bounds, untouched tails,
RGB/RGBA and uniform colors, normals, lighting, camera submission, static caching,
ten water/city resets without new fields or VBOs, source replacement and growth,
and the pinned wheel/core compatibility adapter. Steady-frame checks forbid the
stock staging helpers, downloads and new device VBOs. The Vulkan command runs
these same contracts with CPU fallback disabled; it still does not render pixels.

For a native comparison, run each case in a fresh process on the Intel desktop:

```sh
for scenario in paused active wave; do
  for staging in stock device; do
    python -m fluids.render_benchmark --staging "$staging" --scenario "$scenario" \
      --quality medium --warmup 8 --seconds 30 --resets 10 \
      --gpu 'Intel UHD Graphics CML GT2 (8086:9b41)' \
      --driver 'i915 / Mesa 25.2.8' --refresh-hz 59.96 \
      --capture "${staging}-${scenario}.png" --output "${staging}-${scenario}.json"
  done
done
```

Use the actual installed driver and refresh rate. Repeat in reverse order to
check drift, and repeat with `--quality low` if needed. Keep the same desktop,
1440 × 900, vsync, camera and settings; leave controls untouched while sampling.
`stock` reproduces the original public SceneV2 path without allocating device
VBOs. `paused` stops physics, `active` runs without a wave, and `wave` launches
one default wave after physics compilation. The unchanged application loop is
timed with instance wrappers. First-frame compilation and image capture, plus
eight seconds of warm-up, are excluded. No additional synchronization is added
to sampled frames. Timing is wall-clock FPS, mean frame, render submission and
physics advance time, not isolated GPU timestamps. Reports include simulated
and discarded time, process RSS at sample boundaries, staging metadata, and ten
subsequent water/city reset timings and RSS samples. Device reports assert the
actual draw metadata and active transfer bounds. Core buffers can grow to the
largest scene seen; watch for continued memory growth after that point.

Process RSS is not GPU memory, and VBO sizes are not measured total memory.
Record driver/process GPU memory separately on the desktop, before/after warm-up
and across resets, together with evidence of Intel render-engine activity.
Compare the initial paused stock/device captures for water coverage/color,
floor/building normals and lighting, outlines, and camera framing. The captures
read back pixels only in the excluded first frame. Also complete the manual
interaction/reset pass below; static screenshots do not establish interaction
acceptance. Report paused rendering gains separately from active physics costs.

This implementation environment has Python 3.11.2 and the pinned Taichi wheel,
but no `/dev/dri`, Vulkan ICD or `DISPLAY`. The benchmark fails at Vulkan
initialization without fallback. Intel Vulkan compatibility and the manual
rendered comparison are **unexecuted**; target-GPU paused/active/wave FPS,
submission time, memory and native ten-reset measurements are **unmeasured**.
CPU contracts support the code path but cannot complete hardware acceptance.

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
5. Set high quality with 4 m depth and reset to exercise the largest supported
   initial lattice; tighter bins now fit it without overflow. The smoke check
   forces overflow by collapsing particles into one bin, verifies failure leaves
   fluid state untouched, and tests reset recovery. For startup failure handling, launch with
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

### Issue #38 timestep readbacks

`DynamicsSolver.stable_time_step` now lazily reads and validates the scalar GPU
limit once per successful prepared force snapshot. Integration checks its dt
against that cached bound. Reset, wave launch, every subsequent force preparation
(including failed preparation), and successful integration invalidate the cache.
Force-only callers do not read a timestep. Particle arrays stay on the device.
The grid overflow read still gates all particle-writing physics passes.

The kernel tests count actual scalar field accesses: a complete substep uses
one overflow read and one timestep read, rather than one overflow and two
timestep reads. A saturated 12-step frame therefore has **24 instead of 36**
such reads. The cached and duplicate control also produce the same simulation
progress and particle state in that test. Frames that prepare a snapshot but
lack accumulated time to integrate it can have additional preparation reads;
the benchmark reports observed counts rather than assuming two per substep.

Use the existing benchmark's default fixed 1.0 m grid for the comparison, with
both modes in fresh processes. `duplicate` restores property readback on every
access using the current kernels; it is a readback control, not an old revision.
On the Intel desktop, alternate the order over several repeats:

```sh
for reads in duplicate cached; do
  python -m fluids.benchmark --backend vulkan --timestep-reads "$reads" --steps 40
  python -m fluids.benchmark --backend vulkan --timestep-reads "$reads" \
    --steps 40 --wave --presteps 120
  for scenario in idle wave paused; do
    python -m fluids.benchmark --mode native --timestep-reads "$reads" \
      --scenario "$scenario" --warmup-seconds 8 --sample-seconds 30 \
      --device-label 'Intel UHD Graphics (CML GT2, 8086:9b41)' \
      --driver 'Mesa 25.2.8 / i915' --refresh-hz 59.96
  done
  python -m fluids.benchmark --mode native --timestep-reads "$reads" \
    --quality low --scenario wave --warmup-seconds 8 --sample-seconds 30 \
    --device-label 'Intel UHD Graphics (CML GT2, 8086:9b41)' \
    --driver 'Mesa 25.2.8 / i915' --refresh-hz 59.96
done
```

Record actual driver/refresh metadata and verify i915 render-engine activity on
the target machine. Stage reports include grid/density/forces/timestep-readback/
integration/collision timings and scalar read counts. The timestep stage is
per read (40 samples for cached, 80 for duplicate), not summed per substep.
These diagnostics synchronize each stage and include host dispatch and
synchronization costs; they are not isolated GPU timestamps. Native reports
add no synchronization and include warmed frame and physics-advance timings,
simulation progress per wall second, discarded time, completed substeps, and
scalar read counts. Compare simulation progress as well as frame duration.
Candidate-array downloads occur only outside stage sampling, never in native
sampling or normal simulation.

This run container has no `/dev/dri`, Vulkan ICD, or `DISPLAY`. Intel Vulkan
frame/FPS, simulation-progress, stage-timing, and GPU-memory results are
**unmeasured**; the commands above remain owner measurements. CPU kernel tests
and `--backend cpu` stage diagnostics validate read counts and behavior only.

Device-side overflow/timestep consumption was evaluated but is not enabled in
this change. The existing neighbor readiness guard suppresses traversal, not
the density/pressure writes surrounding it; force, integration, and collision
writes would also need whole-pass guards before any host overflow gate could
be removed. A safe asynchronous path would require ordered guarded passes and
a combined status/timestep result before committing simulation time. Consuming
dt only on the device still leaves the host accumulator needing the actual
completed dt. Retaining the overflow gate preserves failure atomicity and
pause/reset recovery while eliminating the redundant synchronization now.

### Issue #36 grid benchmark

The default grid now uses 1.0 m cells, the largest supported smoothing radius.
All neighbor traversal, including density and forces, uses the grid's layout.
The allocation still covers 48 × 16 × 32 m from the current world's lower bound
and is reused across depth, quality, and city resets. The 256-index cell capacity,
overflow gate, physics settings, and timestep limits are unchanged.

Run each layout in a separate process, alternating order over several repeats:

```sh
for layout in legacy fixed quality; do
  python -m fluids.benchmark --backend cpu --layout "$layout" --steps 40
  python -m fluids.benchmark --backend cpu --layout "$layout" --steps 40 --wave --presteps 120
done
```

`legacy` reproduces the 1.6 m **layout using the updated shared kernels**;
it does not time the old revision's implementation. `fixed` uses 1.0 m cells;
`quality` uses the selected quality's smoothing radius (1.0/0.7/0.5 m).
`--quality` and `--depth` select the scene without changing other physics settings.
Smaller layouts expand their search stencil if a later quality reset increases
the radius; their field allocations stay fixed. The default remains 1.0 m to
limit memory while supporting every quality with a 27-cell search. Medium
quality-sized bins require nearly three times its bin memory; high requires
192.75 MiB. Compact cell ranges would be a separate implementation if those
layouts' target-GPU gains justify adopting them.

Stage mode compiles all kernels, restores the scene, optionally launches a wave
and advances `--presteps`, then warms eight evolving steps before sampling 40.
It reports candidate checks before distance rejection, grid field bytes, and
grid/density/forces/integration/collisions wall-time distributions. Each stage
includes explicit synchronization overhead; diagnostic substep times are not
native frame times. Field downloads for candidate counting happen outside
measured intervals. Evolving scenes are compared independently, not bitwise.

On the Intel CML GT2 desktop, first run `python -m fluids.smoke`, then repeat
the stage commands with `--backend vulkan`. To measure complete native frames
using the existing application loop, run:

```sh
for layout in legacy fixed quality; do
  for scenario in idle wave paused; do
    python -m fluids.benchmark --mode native --layout "$layout" --scenario "$scenario" \
      --device-label 'Intel UHD Graphics (CML GT2, 8086:9b41)' \
      --driver 'Mesa 25.2.8 / i915' --refresh-hz 59.96
  done
done
```

Use the machine's actual driver and refresh rate. Avoid input or settings edits
during a sample. Native mode requires Vulkan without fallback, preserves the
1440 × 900 vsynced window, excludes compilation and the first presented frame,
warms for eight seconds, and samples for at least 30 seconds before closing.
It reports FPS, complete frame durations, physics-advance wall time, completed
substeps, simulated time, discarded time/rate, and accumulator endpoints.
It adds no synchronization to the application loop: physics-advance timings
include its dispatch/readback waits; the complete frames include presentation.
Native and synchronized stage results should be collected separately.
Driver memory still needs owner observation; reported grid bytes are logical
field sizes, excluding backend padding and driver allocations.

Container measurements on 2026-10-12: Debian 12, Python 3.11.2, Taichi 1.7.4,
CPU backend on an advertised AMD Ryzen 7 8745HS; default medium, 18,360 particles.
Sequential single runs, with no tests running concurrently:

| Layout | Initial checks/pass | Checks/particle | Logical grid MiB |
| --- | ---: | ---: | ---: |
| Legacy 1.6 m | 15,520,800 | 845.4 | 5.88 |
| Fixed 1.0 m | 6,238,200 | 339.8 | 24.09 |
| Quality 0.7 m | 2,654,400 | 144.6 | 71.57 |

The fixed layout reduces initial candidate checks by 59.8%. CPU stage medians
(milliseconds, 40 evolving samples after warm-up):

| Start/layout | Grid | Density | Forces | Integration | Collisions | Diagnostic substep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial/1.6 m | 0.26 | 4.67 | 5.32 | 0.13 | 0.36 | 11.07 |
| Initial/1.0 m | 0.25 | 2.41 | 2.75 | 0.12 | 0.28 | 6.06 |
| Initial/0.7 m | 0.27 | 1.87 | 2.21 | 0.10 | 0.22 | 4.79 |
| 120 wave steps/1.6 m | 0.28 | 4.93 | 5.60 | 0.13 | 0.42 | 11.44 |
| 120 wave steps/1.0 m | 0.23 | 3.06 | 3.37 | 0.11 | 0.28 | 7.39 |
| 120 wave steps/0.7 m | 0.32 | 2.93 | 3.24 | 0.11 | 0.38 | 7.01 |

These are container CPU observations, not predicted Intel timings or FPS.
CPU contracts passed (158 tests, one desktop test skipped), including full
neighbor-set comparisons for sampled lattice particles at every quality and
0.5/2/4 m depths, boundary pairs, density/force reference tolerances, collisions,
overflow atomicity, and reset field reuse. The separate CPU diagnostic also
passed 120 wave steps, collision probes, ten resets, and overflow recovery.
Vulkan smoke and native benchmark
attempts both failed to create a Vulkan instance: this container exposes no
GPU device, Vulkan ICD, or `DISPLAY`. Target Intel GPU stage timings, native
frames/FPS, simulated/discarded time trends, driver memory, and visual acceptance
are **unmeasured** and remain owner checks before merging.

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
  constraints. Each of the 48 × 16 × 32 spatial bins holds 256 indices, using
  about 24.09 MiB for bins and their metadata. Every supported initial lattice
  fits these 1.0 m bins (at most 64 particles per cell); compression can still
  overflow a cell and stop physics before any particle fields are changed.
- Physics uses adaptive timesteps between 10 µs and 1/120 s, accepts at most
  0.1 s of each frame and runs at most 16 substeps. Excess requested time is
  discarded rather than integrated in an unsafe large step. High quality,
  deep water, fast waves, high viscosity, or fast playback can reduce effective
  simulation speed. Real interaction and performance depend on GPU and driver.
