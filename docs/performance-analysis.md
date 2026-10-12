# Issue #47: where physics time goes

Analysis date: 2026-10-12. This document describes the checked-out implementation
and measurements made with the existing `fluids.benchmark` stage mode. It changes
no simulation, defaults, panel, benchmark code, or flags.

The owner reports the following **Intel CML GT2 desktop observations**, supplied
in issue #47 rather than measured here: before optimization, 2.25 FPS, about
445 ms mean frame time and 410 ms physics advance, with 27.1 FPS while paused;
currently about 27 FPS with the panel reporting 0.230 simulated s/wall s. The
owner reports unchanged simulation throughput. Improved presentation cadence
does not establish faster physics: the current coordinator limits work and
discards unmet simulation time. These observations do not identify a stage
hotspot or demonstrate that synchronization itself wastes 12 ms.

**All new numbers below are container CPU numbers. Target-GPU stage times,
current per-frame physics time, synchronization overhead, frames/FPS, and driver
memory are unmeasured.** This Debian 12 container exposes neither `/dev/dri`, a
Vulkan ICD directory, nor `DISPLAY`. No native frame measurement was attempted
or inferred. The GPU and CPU can have different bottlenecks.

## 1. How long does the physics loop take?

### What was measured

The environment was Python 3.11.2, NumPy 1.26.4, Taichi 1.7.4 (`Arch.x64`),
Linux 7.0.0-38-generic with glibc 2.36. `lscpu` advertises an AMD Ryzen 7 8745HS
and nine online virtual CPUs under KVM. Taichi used its default CPU threading;
there was no affinity pinning. Offline caching was enabled. Processes ran
sequentially, with no tests running concurrently; host contention was not
controlled. These are observations of this virtualized container, not a CPU
product benchmark.

There were 26 fresh-process stage runs: all nine quality/depth combinations
(low/medium/high, 0.5/2/4 m), each without a wave and with a wave; two additional
repeats of both default scenes; and default-scene controls using quality-sized
cells and duplicate timestep reads. Every run sampled 40 evolving substeps
after eight warm-up steps. Wave runs launched the default impulse and advanced
120 additional steps before warm-up. A wave is a one-time velocity impulse,
not a recurring emitter. The initial compilation wave and step are followed by
a reset and are excluded from sampling.

The following table combines the three **container CPU medium/2 m, fixed 1 m
cell, cached-read runs**. Each mean is the average of the three run means
(equivalently a pooled mean with equal sample counts); each median is the median
of the three run medians, not a pooled sample median. Times are milliseconds.

| Boundary | No wave mean | No wave median | Evolved wave mean | Evolved wave median |
| --- | ---: | ---: | ---: | ---: |
| Grid build (clear + insert + overflow read) | 0.245 | 0.212 | 0.453 | 0.247 |
| Density | 3.158 | 2.477 | 4.715 | 3.064 |
| Forces + dt reduction | 3.794 | 2.824 | 5.046 | 3.434 |
| Drained timestep read (per read) | 0.033 | 0.028 | 0.043 | 0.027 |
| Integration + colors | 0.137 | 0.115 | 0.181 | 0.112 |
| Collisions | 0.320 | 0.276 | 0.416 | 0.271 |
| Whole diagnostic substep | 7.780 | 6.110 | 10.957 | 7.479 |

The default diagnostic substep median ranged from 6.042 to 6.266 ms without
a wave and 7.017 to 11.575 ms with an evolved wave. Per-run substep p95 ranged
from 12.916 to 17.605 ms and from 16.403 to 28.027 ms respectively. Reported
means therefore exceed medians substantially. Small differences between single
control runs below are not established causal speedups.

There is **no container per-frame physics or frame/FPS result**. Stage mode calls
`Simulation.step()`, which takes exactly one adaptive step and ignores the wall
budget, playback speed, pause state, and frame accumulator. It does not call
`Simulation.advance()` or render. Dividing these times into a nominal budget
would not measure native completed steps/frame.

### What the stage boundaries include and omit

The source is [measure_stages](../fluids/benchmark.py). Each wrapped operation
is preceded by `ti.sync()`, timed on the host, and followed by another `ti.sync()`
inside its timer. Thus each stage includes its Python/API dispatch, device work,
and ending completion wait. The preceding drain is excluded from that stage
timer but included in the diagnostic substep timer. In cached mode there are
six wrappers, hence twelve explicit wrapper synchronizations plus one final
synchronization per sampled `step()`. Those diagnostic waits are additional to
the production scalar gates; they are not the production budget's one end-step
wait. Taichi may lower one entry kernel into multiple device dispatches.

| Requested component | Existing measurement boundary | Limitation |
| --- | --- | --- |
| Grid clear and insert | `grid.build`: validation, `_clear`, `_insert`, overflow scalar read, end sync | Clear, insertion, and overflow readback cannot be separated. Clear resets counts, not all 256 index slots. |
| Density | `density._compute` plus end sync | Includes neighbor traversal and EOS, dispatch, and waiting. |
| Forces | `_compute_forces` plus end sync | Includes second traversal, pressure/viscosity/gravity, acceleration cap, and global dt reduction. |
| Scalar timestep readback | `_read_stable_time_step` plus end sync | Prior force work has already been drained in stage mode. This is a drained-queue read boundary, not the native force completion wait. |
| Integration | `_integrate` plus end sync | Includes velocity/position updates and splash colors. |
| Collisions | `collisions.resolve` plus end sync | Includes host radius/world checks and contact kernel; no ordinary per-step geometry upload. |
| End-step budget `ti.sync()` | Not separately measured | Stage `step()` does not invoke the budgeted `advance()` path. |
| Dispatch versus device execution versus idle/wait overhead | Not separately measured | Host synchronized stage times are not GPU timestamps. |

The difference between the sum of stage **means** and the diagnostic substep
mean is about 0.092 ms without a wave and 0.102 ms with a wave in the three-run
CPU aggregate. It includes Python coordination, validation, preceding drains,
and the final drain. It is not a measurement of pure synchronization overhead.
Subtracting stage medians would be invalid because they need not come from the
same substep.

Successful sampled cached substeps all made exactly one overflow and one
timestep field read (40 + 40 reads/run). Duplicate controls made one overflow
and two timestep reads (40 + 80). In the first cached default runs the drained
timestep-read medians were 0.031 ms without a wave and 0.023 ms with a wave;
duplicate controls measured 0.018 and 0.023 ms **per read**, with 80 rather than
40 timed read samples. Counting two reads doubles the number of that boundary's
samples; it does not mean the reported stage mean is already a per-substep sum.
These CPU controls do not establish a GPU readback saving.

### Is the 12 ms spent computing or waiting?

**Unresolved on the target GPU with the available measurements.** The
production path in [Simulation.advance](../fluids/simulation.py) is:

1. Clear/insert, then read overflow before any particle-writing physics pass.
2. Density and force kernels, then read the scalar stable dt. The cached property
   supplies integration's validation without another read.
3. Integrate and collide, then `ti.sync()` when the work budget is enabled.
4. Check elapsed host time against 12 ms and either start another complete step
   or discard the remaining requested time.

The scalar reads require their results to become host-visible. The timestep
gate waits for the force snapshot; the final budget synchronization accounts
for the subsequently queued integration and collision work. Time blocked in
these APIs may be time during which the device is doing useful computation.
It can also include submission, driver, queue, transfer, and wake-up overhead.
A GPU queue can include preceding rendering work, especially at the first
preparation of a native frame. A long host wait is not proof of an idle GPU.

The budget includes all these costs, as well as Python work. It is checked
between completed steps, not a hard deadline or preemption mechanism. One
indivisible step can overrun it; preparation can also occur without integration
when accumulated time is insufficient for the derived dt. Such a snapshot is
cached for reuse. The benchmark reports actual reads rather than assuming two
reads times the number of completed steps in every native frame.

The marginal cost of the budget's `ti.sync()` is **unmeasured**: its total blocked
time would mix necessary integration/collision completion with synchronization
overhead. Native `--physics-budget-ms 0` is an existing useful control, but it
also disables the wall deadline, allows up to sixteen steps, and changes time
discarding. Comparing 0 and 12 measures the scheduler policy together with its
waits, not a clean experiment removing just one synchronization.

If target results still cannot distinguish the causes, a follow-up profiling
proposal should record device timestamps around clear, insert, density, forces,
integration, and collision dispatches; separately time host submission, overflow
read, timestep read, and final sync; and measure an already-drained sync baseline.
Correlate them with GPU busy/queue information on the owner machine. That would
separate work from submission gaps and synchronization latency. No such profiling
code or profiler flags were added here.

### Exact owner commands (target results remain unmeasured)

Run from this repository in its installed Python environment on the Intel
desktop. Unset `TI_ARCH` if present. First verify `python3 -m fluids.smoke`.
Use the actual driver and refresh rate; the refresh value below is the prior
README's owner value and must be checked. The driver placeholder is metadata
to replace before running. Do not edit settings or interact during samples.

```sh
export FLUIDS_GPU_MODEL='Intel CML GT2'
export FLUIDS_GPU_DRIVER='REPLACE_WITH_ACTUAL_DRIVER_VERSION'
export FLUIDS_REFRESH_HZ=59.96
mkdir -p /tmp/fluids-47-target
python3 -m fluids.smoke || exit 1

# Stage matrix: three repeats, reverse quality/depth order on repeat 2.
for repeat in 1 2 3; do
  qualities='low medium high'
  depths='0.5 2 4'
  phases='rest wave'
  if [ "$repeat" = 2 ]; then
    qualities='high medium low'
    depths='4 2 0.5'
    phases='wave rest'
  fi
  for quality in $qualities; do
    for depth in $depths; do
      for phase in $phases; do
        wave_args=()
        if [ "$phase" = wave ]; then wave_args=(--wave --presteps 120); fi
        python3 -m fluids.benchmark --mode stages --backend vulkan \
          --layout fixed --timestep-reads cached --quality "$quality" --depth "$depth" \
          --steps 40 --warmup-steps 8 "${wave_args[@]}" \
          --device-label "$FLUIDS_GPU_MODEL" --driver "$FLUIDS_GPU_DRIVER" \
          --refresh-hz "$FLUIDS_REFRESH_HZ" \
          > "/tmp/fluids-47-target/stages-$quality-$depth-$phase-$repeat.log" || exit 1
      done
    done
  done

  # Existing layout and readback controls; alternate their order.
  layouts='fixed quality legacy'
  reads_modes='cached duplicate'
  if [ "$repeat" = 2 ]; then layouts='legacy quality fixed'; reads_modes='duplicate cached'; fi
  for phase in $phases; do
    wave_args=()
    if [ "$phase" = wave ]; then wave_args=(--wave --presteps 120); fi
    for layout in $layouts; do
      python3 -m fluids.benchmark --mode stages --backend vulkan \
        --quality medium --depth 2 --layout "$layout" --timestep-reads cached \
        --steps 40 --warmup-steps 8 "${wave_args[@]}" \
        --device-label "$FLUIDS_GPU_MODEL" --driver "$FLUIDS_GPU_DRIVER" \
        --refresh-hz "$FLUIDS_REFRESH_HZ" \
        > "/tmp/fluids-47-target/layout-$layout-$phase-$repeat.log" || exit 1
    done
    for reads in $reads_modes; do
      python3 -m fluids.benchmark --mode stages --backend vulkan \
        --quality medium --depth 2 --layout fixed --timestep-reads "$reads" \
        --steps 40 --warmup-steps 8 "${wave_args[@]}" \
        --device-label "$FLUIDS_GPU_MODEL" --driver "$FLUIDS_GPU_DRIVER" \
        --refresh-hz "$FLUIDS_REFRESH_HZ" \
        > "/tmp/fluids-47-target/reads-$reads-$phase-$repeat.log" || exit 1
    done
  done
done
```

The shell examples use Bash arrays so the wave flags remain distinct arguments.
Taichi prints startup lines before the JSON report; these files are logs, not
bare JSON. Record stage mean/median/p95, diagnostic substep, candidates, logical
grid bytes, scalar read counts, and `simulated_seconds / 40` as the sampled mean
dt. `legacy` is the 1.6 m layout with current kernels, not an old revision.

Measure **native per-frame physics separately**, with the existing native mode:

```sh
for repeat in 1 2 3; do
  budgets='12 0'
  reads_modes='cached duplicate'
  if [ "$repeat" = 2 ]; then budgets='0 12'; reads_modes='duplicate cached'; fi
  for budget in $budgets; do
    for reads in $reads_modes; do
      for scenario in idle wave paused; do
        python3 -m fluids.benchmark --mode native --backend vulkan \
          --quality medium --depth 2 --layout fixed --timestep-reads "$reads" \
          --physics-budget-ms "$budget" --scenario "$scenario" \
          --warmup-seconds 8 --sample-seconds 30 \
          --device-label "$FLUIDS_GPU_MODEL" --driver "$FLUIDS_GPU_DRIVER" \
          --refresh-hz "$FLUIDS_REFRESH_HZ" \
          > "/tmp/fluids-47-target/native-$scenario-$budget-$reads-$repeat.log" || exit 1
      done
    done
  done
done

# Native quality/depth check under the current scheduling policy.
for quality in low medium high; do
  for depth in 0.5 2 4; do
    python3 -m fluids.benchmark --mode native --backend vulkan \
      --quality "$quality" --depth "$depth" --layout fixed --timestep-reads cached \
      --physics-budget-ms 12 --scenario wave --warmup-seconds 8 --sample-seconds 30 \
      --device-label "$FLUIDS_GPU_MODEL" --driver "$FLUIDS_GPU_DRIVER" \
      --refresh-hz "$FLUIDS_REFRESH_HZ" \
      > "/tmp/fluids-47-target/native-wave-$quality-$depth.log" || exit 1
  done
done
```

Native mode uses the application's 1440 × 900 vsynced window, excludes compilation
and the first presented frame, then warms for eight wall seconds. Report
`physics_advance_wall`, complete `frame`, FPS, `steps_per_frame`, budget exhaustion
and overrun distributions, simulated/discarded s per wall s, accumulator endpoints,
and scalar reads. Paused is a rendering/control baseline; subtracting paused frame
time from running frame time is not an isolated physics measurement. Native mode
adds no timing synchronization beyond the selected production coordinator policy.
Its synthetic input probes measure scheduled dispatch/presentation return latency,
not physical input-to-photon latency. The native wave warm-up is wall-based,
whereas stage presteps are step-based, so the two sample different physical ages.

## 2. Where are the hotspots, and how do they scale?

For **container CPU medium/2 m**, density plus forces account for 89.4% of the
three-run mean diagnostic substep without a wave and 89.1% with an evolved wave.
Forces are slightly more expensive than density in the default samples. They
both repeatedly scan the same cell stencil and candidate indices; forces add
square roots, velocity/density/pressure loads, viscosity, and timestep reduction.
The grid, integration, and collisions are secondary there. Target-GPU ranking
is unmeasured, including whether compute, memory access, dispatch, or waits dominate.

The complete **container CPU matrix** follows. These are first-run stage medians,
in ms; `step` is the independently measured diagnostic substep median. They are
not additive and are not native frame times. `rest` means no launched wave, not
a static scene: gravity, pressure, and contacts still evolve it. `wave` samples
after 120 presteps plus eight warm-up steps. Only medium/2 m has repeated runs;
the other rows establish observed scaling patterns rather than precise ratios.

| Quality / depth (m) | Phase | Grid | Density | Forces | dt read | Integration | Collisions | Step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low / 0.5 | rest | 0.154 | 0.133 | 0.169 | 0.014 | 0.054 | 0.173 | 0.802 |
| low / 0.5 | wave | 0.152 | 0.243 | 0.265 | 0.016 | 0.051 | 0.176 | 0.977 |
| low / 2 | rest | 0.155 | 0.601 | 0.774 | 0.016 | 0.070 | 0.180 | 1.906 |
| low / 2 | wave | 0.179 | 0.992 | 1.172 | 0.019 | 0.079 | 0.207 | 2.739 |
| low / 4 | rest | 0.190 | 1.409 | 1.715 | 0.023 | 0.096 | 0.217 | 3.913 |
| low / 4 | wave | 0.215 | 2.401 | 2.621 | 0.033 | 0.109 | 0.264 | 5.757 |
| medium / 0.5 | rest | 0.148 | 0.233 | 0.263 | 0.014 | 0.062 | 0.177 | 1.003 |
| medium / 0.5 | wave | 0.166 | 0.302 | 0.332 | 0.016 | 0.067 | 0.186 | 1.162 |
| medium / 2 | rest | 0.206 | 2.477 | 2.839 | 0.031 | 0.122 | 0.289 | 6.266 |
| medium / 2 | wave | 0.197 | 2.994 | 3.356 | 0.023 | 0.107 | 0.253 | 7.017 |
| medium / 4 | rest | 0.327 | 6.860 | 8.779 | 0.063 | 0.228 | 0.500 | 17.629 |
| medium / 4 | wave | 0.360 | 8.361 | 9.325 | 0.057 | 0.196 | 0.463 | 19.443 |
| high / 0.5 | rest | 0.228 | 1.419 | 1.700 | 0.030 | 0.111 | 0.297 | 4.039 |
| high / 0.5 | wave | 0.201 | 2.234 | 2.412 | 0.026 | 0.104 | 0.235 | 5.291 |
| high / 2 | rest | 0.299 | 15.815 | 18.647 | 0.067 | 0.285 | 0.516 | 37.487 |
| high / 2 | wave | 0.358 | 18.597 | 20.521 | 0.072 | 0.298 | 0.589 | 44.002 |
| high / 4 | rest | 0.376 | 65.192 | 65.145 | 0.091 | 0.441 | 0.637 | 133.460 |
| high / 4 | wave | 0.429 | 66.299 | 76.155 | 0.091 | 0.435 | 0.707 | 164.626 |

The sampled mean dt was 8.333 ms for every low and medium row in both phases.
For high at 0.5/2/4 m it was 8.333/8.188/8.026 ms without a wave and
6.960/6.882/5.653 ms with an evolved wave. These values come from reported
simulated seconds divided by 40, not a new measurement hook. The benchmark
does not report the winning timestep constraint or a dt distribution. It cannot
prove that an acoustic limit, acceleration, displacement, or viscosity caused
these smaller high-quality steps.

Candidate counts explain part of the cost structure. They count ordered
particle pairs in the stencil **before radius rejection, including self**, for
one pass. Density and forces each traverse those candidates, so two such scans
occur per ordinary substep. The benchmark records initial and sample-start
snapshots; it does not average candidate counts over all timed steps.

| Quality / depth (m) | N | Initial lattice checks/pass | Initial checks/particle | Wave sample-start checks/pass | Wave checks/particle |
| --- | ---: | ---: | ---: | ---: | ---: |
| low / 0.5 | 1,739 | 58,149 | 33.4 | 64,365 | 37.0 |
| low / 2 | 6,956 | 930,384 | 133.8 | 785,588 | 112.9 |
| low / 4 | 13,912 | 2,325,960 | 167.2 | 2,081,168 | 149.6 |
| medium / 0.5 | 3,672 | 249,528 | 68.0 | 273,376 | 74.4 |
| medium / 2 | 18,360 | 6,238,200 | 339.8 | 5,063,958 | 275.8 |
| medium / 4 | 40,392 | 18,215,544 | 451.0 | 17,212,938 | 426.1 |
| high / 0.5 | 14,250 | 3,823,300 | 268.3 | 4,416,310 | 309.9 |
| high / 2 | 57,000 | 61,172,800 | 1073.2 | 52,137,182 | 914.7 |
| high / 4 | 114,000 | 152,932,000 | 1341.5 | 140,605,376 | 1233.4 |

With fixed 1 m cells and roughly uniform volumetric packing, candidates per
particle grow with local number density (approximately spacing to the power
-3), even though the true kernel radius shrinks with spacing. At fixed quality,
adding depth adds particles and initially fills more of each vertical stencil;
candidate work can grow faster than particle count in shallow scenes. For
example, going from medium/0.5 m to medium/2 m multiplies particles by five but
initial candidates by twenty-five. Going from medium/2 m to high/2 m multiplies
particles by about 3.10 and initial candidates by about 9.81. These are scene
counts, not GPU speedups. Once occupancy/stencil geometry is comparable,
density and force work is approximately O(N × candidates per particle), not
necessarily O(N), and pathological compression can make the grid unusable.

Grid work is O(number of allocated cells + N) plus atomic contention and a
scalar gate. Integration is O(N). Collision work is O(N) with bounded checks:
container faces, land, and at most one building plot per particle, rather than
a scan of all buildings. The CPU matrix follows these broad distinctions:
high-quality density/forces grow much more than integration/collisions. The
low/0.5 m row is small enough that fixed dispatch/control/contact costs are a
large fraction, so the default 89% neighbor-pass share is not universal.

An active wave adds no continuous wave kernel to ordinary substeps after launch.
It changes velocities, density, accepted-neighbor counts, locality, contacts,
and potentially dt. In the default first run, candidate checks at sample start
fell from 6,238,200 to 5,063,958, yet density and force medians rose. A lower
candidate snapshot therefore does not prove a cheaper evolving pass; accepted
neighbors, spatial distribution, subsequent motion, and execution variability
also matter. Different qualities have different mean dt, so 128 presteps do
not necessarily place all wave scenes at the same simulated age.

The existing quality-sized-cell controls provide another **container CPU**
observation, not a default change:

| Phase / cell width | Logical grid MiB | Initial checks/pass | Sample-start checks/pass | Density median ms | Forces median ms | Step median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rest / 1.0 m | 24.09 | 6,238,200 | 6,238,200 | 2.477 | 2.839 | 6.266 |
| rest / 0.7 m | 71.57 | 2,654,400 | 2,136,128 | 1.951 | 2.283 | 5.041 |
| wave / 1.0 m | 24.09 | 6,238,200 | 5,063,958 | 2.994 | 3.356 | 7.017 |
| wave / 0.7 m | 71.57 | 2,654,400 | 2,098,370 | 3.092 | 3.384 | 7.352 |

At medium, 0.7 m cells reduce initial candidates by 57.4%, but logical grid
storage rises from 24.09 to 71.57 MiB. The initial diagnostic median decreased
from 6.266 to 5.041 ms in these single runs; the evolved-wave control did not
improve it (7.017 versus 7.352 ms). No target gain is established. At high,
the existing quality layout uses 0.5 m cells and 192.75 MiB of logical grid
storage; it was not timed here. The owner commands can repeat the matrix with
`--layout quality` to evaluate that tradeoff. These bytes exclude padding and
driver allocations.

For reproducibility, the CPU matrix used the stage-matrix commands in section 1
with `--backend cpu`, the container device label, and no refresh argument; one
full matrix, followed by the two extra medium repeats (wave/rest, then rest/wave).
Controls ran at medium/2 m in order rest-quality, rest-duplicate, wave-quality,
wave-duplicate. Every other setting retained its current default. A minimal
pair reproducing the main scene is:

```sh
python3 -m fluids.benchmark --mode stages --backend cpu --layout fixed \
  --timestep-reads cached --quality medium --depth 2 --steps 40 --warmup-steps 8
python3 -m fluids.benchmark --mode stages --backend cpu --layout fixed \
  --timestep-reads cached --quality medium --depth 2 --steps 40 --warmup-steps 8 \
  --wave --presteps 120
```

## 3. What physics are we running?

### Solver, kernels, and neighbor storage

This is a **weakly compressible SPH approximation with a linear, clipped EOS**,
not an incompressible pressure projection or the usual stiff seventh-power
Tait EOS. [DensitySolver](../fluids/density.py) computes density from positions
every prepared step, including self:

```text
rho_i = sum_j m W_poly6(|x_i - x_j|, h)
W_poly6(r,h) = 315/(64*pi*h^9) * (h^2-r^2)^3,  0 <= r <= h
p_i = min(400 * max(rho_i - 1000, 0), 400000) Pa
```

The EOS stiffness is 400 m²/s², giving nominal artificial sound speed
sqrt(400) = **20 m/s** in its unclipped positive-pressure range. Pressure is zero
below rest density and saturates at twice rest density (rho = 2000 kg/m³).
The timestep nevertheless retains the nominal 20 m/s term even where pressure
is clamped. Actual estimated densities are stored without a floor; divisions
in forces use `rho_bar = max(rho, 100)` kg/m³. There is no attractive negative
pressure, incompressibility iteration, or physical-water acoustic model.

| Quality | Spacing (m) | Kernel support radius h (m) | Collision sphere radius (m) | Particles at 2 m |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.50 | 1.00 | 0.250 | 6,956 |
| Medium (default) | 0.35 | 0.70 | 0.175 | 18,360 |
| High | 0.25 | 0.50 | 0.125 | 57,000 |

Here h is the **full support radius**, equal to two particle spacings; there is
no additional 2h support. Every kernel is zero outside h. Mass is uniform within
a scene and equals `1000 * requested_water_volume / N`, including edge corrections
from the floored lattice dimensions. The requested 2 m ocean volume is 905.81 m³;
medium particle mass is about 49.336 kg. This preserves initial water quantity
across quality changes; reducing quality changes resolution, not total water mass.

[DynamicsSolver](../fluids/dynamics.py) uses the spiky pressure gradient and
standard viscosity Laplacian. Define `offset = x_j - x_i`, `r = |offset|`,
and `q = 1 - r/h` for r < h:

```text
gradient factor in the implementation = 45/(pi*h^4) * q^2 * offset/r
a_pressure_i = -sum_(j != i) m * (p_i/rho_bar_i^2 + p_j/rho_bar_j^2)
                                    * gradient factor
laplacian_viscosity(r,h) = 45/(pi*h^5) * q
rate_ij = nu * m/rho_bar_j * laplacian_viscosity(r,h)
a_viscosity_i = sum_(j != i) rate_ij * (v_j - v_i)
a_i = a_pressure_i + a_viscosity_i + (0, -gravity, 0)
```

Pressure gradients are suppressed at coincident particles within 1e-6 m;
coincident distinct particles still exchange velocity through viscosity.
The default kinematic viscosity is 0.01 m²/s, not a separate artificial-viscosity
shock term. Pressure pairs are equal and opposite before acceleration limiting;
viscosity weighting, caps, damping, and contacts do not promise exact momentum
or energy conservation. Acceleration magnitude is capped at 1000 m/s² and
integration/wave velocity magnitude at 50 m/s.

[SpatialGrid](../fluids/grid.py) uses fixed **1.0 m cells**, a **27-cell stencil**
(current cell and all offsets -1/0/+1), and **256 index slots per cell**. The
48 × 16 × 32 allocation covers 48 × 16 × 32 m from the current world's lower
bound; the default physical world is x = -20..20, y = 0..12, z = 0..24 m.
Cell coordinates floor positions relative to that origin, with boundary
roundoff clamped. Neighbor cells outside the allocation are skipped, not
clamped into duplicate visits. Each candidate gets a squared-distance support
test. Both density and force kernels perform the traversal separately on the
same rebuilt grid. No neighbor list is retained between passes or steps.

Clear resets cell counts and metadata; insertion atomically reserves a slot.
Old unused indices are harmless because traversal respects the current count.
Counts include overflow attempts, and only slots below 256 are written. The
host overflow result gates **all** density, force, integration, and collision
passes. An overflow leaves particle fields unchanged for that failed substep,
consumes no simulated time, reports an error, discards backlog, and causes the
application to pause. Grid metadata itself changes. Particle capacity is
131,072, independently of cell capacity; the largest supported initial lattice
has 114,000 particles and initial cells hold at most 64. Compression can still
exceed 256. The grid uses 25,264,156 logical bytes (24.09 MiB).

### Integration and adaptive timestep

Forces read one pre-integration position/velocity snapshot. Integration is
explicit **semi-implicit Euler** (velocity first, then position), followed by
contact projection:

```text
v_new = cap_speed((v_old + a_i*dt) * exp(-damping*dt), 50 m/s)
x_new = x_old + v_new*dt
```

Default damping is 0.1/s. Splash colors are updated in the integration kernel
from speed and under-density. No implicit force solve occurs.

The force pass computes per-particle candidate bounds and atomically reduces
them into a global minimum initially set to `MAX_TIME_STEP`. Let `s` be the old
speed capped at 50 m/s, `a` the combined acceleration magnitude capped at
1000 m/s², `h` the quality radius, `travel = 0.4*h/2 = 0.4*spacing`,
and `R_i = sum_j rate_ij`. The candidates are:

| Bound | Formula | Purpose |
| --- | --- | --- |
| Acoustic CFL | `0.4*h / (20 + s)` | Acoustic and advective motion through kernel support. |
| Acceleration | `sqrt(0.4*h / max(a, 1e-6))` | Limit acceleration-driven travel. |
| Semi-implicit displacement | `2*travel / (s + sqrt(s*s + 4*a*travel))` when denominator > 1e-6 | Ensures `s*dt + a*dt*dt <= 0.4*spacing` before contact projection. |
| Viscosity diffusion | `h*h / (6*nu)` when nu > 1e-6 | Explicit spatial diffusion limit. |
| Local viscosity rate | `0.5 / R_i` when R_i > 1e-6 | Bound the summed explicit neighbor relaxation rate. |

The host reads the reduced scalar once per successful force snapshot and clamps
it to **10 µs <= dt <= 1/120 s (8.333 ms)**. Integration checks both these global
bounds and the snapshot stability bound (with a relative 1e-6 tolerance).
The lower clamp is a safeguard, not proof of stability if a raw bound ever falls
below it. The upper cap and non-acoustic limits must remain part of any comparison.

At zero speed the acoustic candidates are 20/14/10 ms for low/medium/high,
all above the 8.333 ms cap. Lowering sound speed alone cannot enlarge a resting
capped step. In sampled medium runs every dt equaled that cap. A different
incompressibility method could improve density behavior, but cannot provide
larger effective timesteps there while keeping the same upper cap. With fast
waves, high quality, strong acceleration, or high viscosity, other bounds can
become restrictive. Nothing in current stage output identifies the active bound.

### Frame scheduling, contacts, and lifecycle contracts

The application uses `Simulation.advance`, rather than the standalone
`DynamicsSolver.advance` convenience method. It accepts at most
**MAX_FRAME_TIME = 0.1 s** of wall time per frame, multiplies accepted time by
playback speed, and tries at most **MAX_SUBSTEPS = 16** complete adaptive steps
within the **12 ms completed-work budget**. Playback scales requested time, not
dt bounds. Excess wall time is counted as discarded simulation time after
playback scaling. Hitting either work or count cap discards all remaining
accumulated time, including tiny remainders. If there is insufficient time for
a complete stable dt and no cap is hit, the fraction carries to the next frame
and the prepared snapshot may be reused. It never enlarges dt to catch up.

[CollisionSolver](../fluids/collisions.py) treats the six closed container faces,
the 0.25 m flat land slab, and procedural buildings as axis-aligned boxes.
Particle centers are clamped to the container inset by their collision radius;
solid boxes are expanded by that radius. Penetrations project to the nearest
accessible face, land first and then at most the particle's current building
plot. Building bottoms buried in the land are excluded. Fixed 6 × 8 plot lookup
is valid because geometry retains at least 0.5 m plot-edge clearance and radius
is at most 0.36 m. Incoming normal velocity is reflected with restitution
(default 0.2), and tangential velocity loses the friction fraction (default 0.1).
Departing velocity is retained. This is conservative discrete projection, not
continuous collision detection or an SPH boundary-particle pressure model.

Reset validates capacity before replacing a scene, reuses particle/grid/contact
allocations, regenerates seeded geometry and the lattice, clears time and errors,
and preserves the user's pause selection. Pause and zero playback request no
integration and discard accumulated backlog; paused wall time is not counted as
discarded requested time. Resuming does not catch up the paused interval.
Single-step takes exactly one complete adaptive timestep irrespective of pause,
zero playback, or work budget, discards the old backlog, and updates time once.
Wave launch is permitted while paused and invalidates prepared forces. Reset,
wave launch, new force preparation (including failure), and successful integration
invalidate the timestep cache.

Determinism needs qualification: seeded geometry/lattice generation is
deterministic, but atomic bin reservation leaves neighbor order unspecified,
and floating-point sums can change with execution order or backend. The
wall-clock budget can change steps per rendered frame with machine load.
Existing tests check identical step-sequence behavior and readback controls in
controlled scenes; they do not establish bitwise replay for every GPU run.
Any optimization should distinguish preserving equations and a prescribed
completed-step sequence from preserving bitwise results or wall-frame progress.

## 4. What could be changed?

The following is an ordered evaluation plan, not an implementation commitment.
**No option has a measured target-GPU speedup.** Numerical factors below are
container observations or explicit algebraic bounds/conditions, never Intel
frame/FPS predictions. For a stage occupying fraction f of completed work, a
k-fold improvement to that stage gives at most `1 / (1 - f + f/k)` overall;
use measured target f, including host/device waits, before making a forecast.

### 1. Evaluate cell sizing and lower quality using existing controls

**Magnitude:** Medium quality-sized cells eliminate 57.4% of initial candidate
checks; the one CPU rest control was about 1.24× faster per diagnostic substep,
while its evolved-wave control was slightly slower. Low/2 m has 6,956 rather
than 18,360 particles (62.1% fewer) and 0.93M rather than 6.24M initial candidates.
The first CPU rest low/medium diagnostic medians were 1.906/6.266 ms. These
differences justify owner comparisons, not a target speedup promise or a default
change. Smaller bins help rejected-candidate work, but can increase grid clear
cost, metadata, and memory traffic.

**Cost and behavior:** Cell-size tuning can preserve the exact neighbor set and
solver equations with a complete stencil, but changes traversal order and float
rounding. Quality-sized medium storage is nearly 3× larger; high is 8× the fixed
grid. Lower quality produces larger visible spheres, a coarser surface/splashes,
different discretization and dt bounds, while preserving initial water quantity.
Changing h independently would alter physics and neighbor count as well.

**Contracts:** Both retain seeded reset behavior, numerical min/max dt, the
overflow gate, and pause/single-step semantics. Bitwise dynamics are not promised
for a cell-layout change and are deliberately different for a quality change.
Reset must retain fixed allocations and either validate an appropriate wider
stencil or rebuild metadata. Do not shrink the 256-slot capacity or silently
truncate lists. Change a default only after target timing and visual acceptance.

### 2. Reuse neighbors: first across passes, then consider Verlet with a skin

**Magnitude:** A per-particle full neighbor list built from the successful grid
could serve density and forces without repeating the 27-cell candidate scan.
Building it every step still requires one scan plus list writes and two neighbor
iterations. If the entire current neighbor-pass work could hypothetically be
halved, the observed CPU default share f ≈ 0.89 gives an algebraic overall ceiling
of about 1.8×; list construction overhead and retained arithmetic reduce that
benefit. This is a conditional work model, not a measured or expected Intel factor.
Expect the largest opportunity where rejection/traversal dominates; little gain
or a regression is possible if accepted-neighbor arithmetic or list bandwidth
dominates.

Verlet lists store neighbors within `h + skin`, use the exact current h test at
each evaluation, and reuse the list until maximum displacement from reference
positions reaches half the skin. This trades fewer rebuilds for more stored
pairs and distance tests; it is the standard approach described in
[LAMMPS's neighbor-list documentation](https://docs.lammps.org/Developer_par_neigh.html).
If lists remain valid for K steps, rebuild work is amortized by K, not the whole
substep. A large skin can erase the saving and require a wider cell stencil.

**Cost and behavior:** Additional fixed capacity/offset/count/reference-position
buffers, safe count/fill passes, and rebuild criteria; memory is approximately
`4*N*L` bytes just for 32-bit neighbor indices with average L entries. Fixed
per-particle limits need their own overflow detection. Correct complete lists
preserve the equations; stale/incomplete lists change pressure, viscosity, and
stability. The rebuild displacement test must include integration **and collision
projection**; speed × elapsed dt alone does not bound projected positions.
Prefer full per-particle lists initially: half-pair lists would require shared
force accumulation/atomics and different viscosity accounting.

**Contracts:** Keep the same timestep reduction and global bounds. Sorted
particle-ID lists can provide reproducible neighbor summation order, but will
not preserve the current unsorted sum bitwise. Reset/quality/radius changes
invalidate lists; wave launch invalidates forces and requires checking reuse
criteria before the next step. Pause performs no autonomous physics; single-step
must validate/rebuild before its one step. Crucially, **the current grid overflow
atomicity is not automatically preserved by skipping grid builds**: occupancy
can exceed 256 during reuse. Retaining that contract requires a cheap current-cell
occupancy check before every particle-writing step (or continued grid builds),
plus whole-step rejection on list overflow. The first across-pass-only version
can retain the existing per-step grid gate unchanged. Multi-step reuse is a
larger design decision with additional validation work.

### 3. Reduce host gates and synchronization only after measuring their share

**Magnitude:** If pure host gate/idle overhead occupies fraction f of target
work, eliminating it entirely has ceiling `1/(1-f)`. At 10% that is 1.11×;
at 50% it is 2×. These are sensitivity examples, not measured shares. Removing
a wait does not remove the device work being waited for. The drained CPU timestep
read is small; it says little about Vulkan driver costs. There is already one
cached timestep read per prepared snapshot, so merely caching it again adds no
saving.

**Cost and behavior:** Consider ordered device-side overflow/status gates and
timestep consumption, then a combined completion result carrying status, actual
dt, and completed-step count. `grid._ready` currently guards neighbor traversal
only: density/pressure writes surrounding it and subsequent force/integration/
collision writes would still execute if their host gate were removed. Every
particle-writing pass needs a whole-pass guard. CPU time bookkeeping must commit
only successful completed work and use the actual device dt; it cannot guess dt
or assume success. Failure-atomic speculative solver iterations may require
scratch buffers.

Reducing the per-substep `ti.sync()` could use completion events or bounded
submission batches, but merely deleting it counts queued integration/contacts
as finished and changes the meaning of the 12 ms budget. Scalar preparation
reads would still serialize much of the loop. A batch must bound outstanding
work and check completion before deciding to submit more; checking only every
B steps permits a different overrun envelope than the current single-step one.

**Contracts:** Preserve global dt bounds with device-side checks and return actual
dt. Fully guarded ordered passes can preserve overflow atomicity and equations,
subject to existing float-order limitations; combining readbacks alone should
not alter seeded reset behavior. Preserve the existing completed-work budget
contract using completion tracking, or explicitly agree on a revised contract
in a follow-up. Pause/reset must drain or cancel pending work safely; reset cannot
race reused allocations. Paused wave launch and single-step need ordered visibility,
and single-step must complete exactly one accepted step before its result is
reported. Asynchrony is therefore medium/high implementation and lifecycle risk,
not a free removal of blocking calls.

### 4. Compact/sort the grid and improve neighbor memory access

**Magnitude:** Fixed storage pays for 256 slots in every allocated cell,
including empty cells. Compact cell ranges plus sorted indices could make
quality-sized bins practical and improve contiguous access. Benefit is bounded
by the target memory/traversal share; there is no measured factor here. Counts,
prefix sums, scatter/sort, and extra dispatches can make small scenes slower.
This is more promising if target density/forces are bandwidth/latency limited
and smaller-bin memory is the obstacle. Fusion of integration and contacts,
or cheaper splash-color updates, targets a much smaller CPU share and should
follow evidence of significant dispatch overhead on the GPU.

**Cost and behavior:** New fixed buffers, prefix-sum/sort machinery, index mapping
and possibly reordered particle arrays. Keep stable particle IDs and restore
reset ordering. Sorted cell contents can improve replay order but change current
float sums; physics need not otherwise change. Color simplification affects
appearance. A fused integrate/contact pass must retain the original update and
projection order.

**Contracts:** Preserve min/max dt and adaptive checks; retain a 256-occupancy
failure gate even if compact storage can hold more, unless an explicit capacity
contract change is agreed. Gate before all particle writes. Reset field reuse,
pause, wave invalidation, and one-step behavior must be tested with reordered
storage. Allocation or sort failure must leave the scene recoverable.

### 5. Decouple rendering from simulation and interpolate completed states

**Magnitude:** Interpolation improves apparent motion between completed steps;
it does not make pressure/force work cheaper or add simulated seconds. If total
time is `P + R` for physics and rendering, removing a fraction q of rendering
cost has ceiling `(P+R)/(P+(1-q)*R)`; P and R must be measured under the intended
GPU scheduling policy. Rendering less often might free device time if rendering
competes with physics. Rendering more often while physics cannot keep up might
consume more of that shared budget. Native rates and latency are unmeasured here.

**Cost and behavior:** Keep previous/current completed positions (and coherent
colors) in device buffers, interpolate only into render storage, and schedule
simulation independently with bounded work. Interpolation adds approximately
one simulation-state interval of visual lag and can blend through collision
projections. Do not extrapolate unvalidated motion through solids. With discarded
time, interpolation fraction must follow the actual completed simulation clock,
not an ever-growing requested-wall-time backlog.

**Contracts:** Rendering must never feed blended positions into physics; then
numerical dt bounds and overflow atomicity can remain intact. Seeded reset and
prescribed step-sequence equations can be preserved, but new cadence changes
wall-frame grouping and discarding. Clear both interpolation endpoints on reset;
freeze and settle the displayed state on pause; show a single-step's completed
state immediately. Independent simulation queues must obey the same drain/
cancellation rules as option 3. This is primarily a presentation/cadence change,
with no guaranteed throughput speedup.

### 6. Evaluate a different incompressibility treatment

[Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf) iteratively
projects predicted particle positions to reduce density error and reconstructs
velocities; [PCISPH](https://www.ifi.uzh.ch/dam/jcr:ffffffff-daa5-74d6-0000-00005a4f5c99/pcisph.pdf)
iterates predicted density and pressure corrections before committing motion.
Both replace the stiff explicit-EOS restriction with iterative work and introduce
new accuracy/convergence choices. Their published gains on other implementations
and hardware are not transferable measurements for this repository.

**Magnitude:** Throughput changes approximately by
`(dt_new/dt_old) * (cost_old_step/cost_new_step)`. A larger dt is useful only if
its ratio exceeds the extra per-step cost. In sampled medium scenes, dt_old is
already 8.333 ms, so preserving the current upper cap leaves no timestep-growth
headroom; an iterative solve may simply cost more. For the high-wave sample mean
dts, the **upper-cap-only** headroom to 8.333 ms is about 1.20×/1.21×/1.47× at
0.5/2/4 m before paying for iterations, and the winning limits are unknown.
Those are arithmetic headroom ratios, not proposed timesteps or target speedups.
Multi-fold throughput gains would require a much more restrictive old bound,
cheaper step work, or an agreed revision of the current timestep cap. Do not
promise a 10× solver gain from the literature.

**Cost and behavior:** Major solver work: predicted-state buffers, density-error
criteria, iteration limits, repeated neighbor/contact passes, convergence/failure
handling, and matching viscosity, free surfaces, and wave behavior. Position
projection can dissipate energy; adding vorticity restoration or artificial
pressure introduces further visual/model choices. PCISPH convergence varies
with the scene. Neighbor validity must cover predictor/correction motion, not
just old positions. Density fidelity may improve, but stiffness, splashes,
contacts, damping response, and energy behavior will differ.

**Contracts:** These are different dynamics and do not preserve the existing
step sequence or bitwise results. Fixed iteration ordering can support reproducible
new dynamics; early stopping introduces further sensitivity. Keep the global
10 µs..8.333 ms bounds initially, retain appropriate displacement/viscosity/contact
limits, and derive solver-specific stability criteria rather than bypassing the
existing guards. Larger-than-cap timesteps explicitly change the timestep contract.
Whole-step overflow atomicity needs scratch state and commit-after-success if a
predictor grid or list overflows after iterations start. Reset must clear/reuse
all new buffers; pause must stop commits; single-step must mean one full accepted
new-solver step, including all iterations, not one iteration. A long iterative
step also increases the indivisible work-budget overrun. This should be a separate
experimental follow-up with numerical and visual acceptance criteria.

Simply lowering the current artificial sound speed is cheaper to implement but
changes pressure stiffness and compressibility, while other bounds and the upper
cap remain. It offers no dt gain in the sampled medium scenes and risks softer,
more compressible water. Raising dt caps, reducing damping, or ignoring viscosity
limits is not an evidence-backed performance fix. Likewise, increasing playback
speed requests more work but does not speed a completed substep; adding backlog
would change responsiveness/discard contracts. None is proposed for this issue.

## What I would do first and why

Collect the target fixed-cell medium/2 m stage and native idle/wave/paused results
first, retaining simulation rate, actual dt, completed steps, discarded time,
and repeat spread alongside FPS. Then compare quality-sized cells and low quality
using the existing commands. These require no code changes and test the strongest
CPU clue: repeated neighbor traversal dominates, while medium timesteps already
reach the cap. If target neighbor passes dominate, start with across-pass neighbor
reuse while retaining the current grid overflow gate. If host/device idle overhead
dominates, request the finer breakdown before designing asynchronous gates.
Defer a solver replacement until target measurements demonstrate timestep
headroom worth more than its extra iterations and changed dynamics.

Validation for this analysis: all 26 CPU stage runs completed successfully; their
reported sample counts, scalar reads, and bounds were checked. The existing
`python -m unittest tests.test_benchmark tests.test_simulation.WorkBudgetTests`
passed 11 tests (benchmark accounting and mocked scheduling contracts, not a GPU
physics validation). Target Vulkan/native/visual validation remains unmeasured.
