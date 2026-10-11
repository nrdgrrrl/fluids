"""Shared settings contract for the UI, scene generator, and simulation.

Lengths are metres and time is seconds. The rectangular domain uses (x, y, z)
coordinates with y pointing up and gravity acting in the negative y direction.
Water depth and building heights are measured from the domain floor. Wave
amplitude is peak displacement from still water; wave direction is measured in
degrees from +x toward +z in the horizontal plane.
SETTINGS_SPECS is the source of defaults, inclusive ranges, units, and reset
policy; consumers should use it instead of defining their own slider bounds.

Settings are immutable and validate on construction. Use ``with_updates`` to
validate a whole proposed edit before applying it. Compare ``reset_signature()``
with the signature of the generated scene to decide whether to rebuild particles
and buildings. Live controls change the existing scene without rebuilding it.

Numerical limits below are solver safeguards, not claims of physical accuracy.
The solver must cap frame time, use adaptive steps within the time-step bounds,
limit substeps, and discard excess accumulated time instead of taking an oversized
step. Playback speed scales elapsed simulation time, never the maximum step size.
"""

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from types import MappingProxyType


DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (40.0, 12.0, 24.0)
DOMAIN_SIZE = tuple(upper - lower for lower, upper in zip(DOMAIN_MIN, DOMAIN_MAX))
DOMAIN_HEIGHT = DOMAIN_SIZE[1]
BOUNDARY_MARGIN = 0.05

# Exactly three particle-spacing levels, in metres. A quality change rebuilds
# particle positions and any solver buffers derived from the spacing.
PARTICLE_SPACING = MappingProxyType({"low": 0.5, "medium": 0.35, "high": 0.25})

NUMERICAL_EPSILON = 1e-6
REST_DENSITY = 1000.0  # kg/m^3
MIN_DENSITY = 0.1 * REST_DENSITY  # floor before density division
SMOOTHING_LENGTH_FACTOR = 2.0  # multiply particle spacing
MIN_TIME_STEP = 1e-5
MAX_TIME_STEP = 1.0 / 120.0
MAX_FRAME_TIME = 0.1
MAX_SUBSTEPS = 16
CFL_FACTOR = 0.4
MAX_SPEED = 50.0  # m/s; solver velocity cap
MAX_ACCELERATION = 1000.0  # m/s^2; cap the magnitude of combined forces


@dataclass(frozen=True)
class SettingSpec:
    """One control's metadata and validation, shared by all consumers."""

    label: str
    group: str
    default: float | int | str
    minimum: float | int | None = None
    maximum: float | int | None = None
    unit: str = ""
    integer: bool = False
    choices: tuple[str, ...] = ()
    requires_reset: bool = False

    def validate(self, name: str, value: object) -> float | int | str:
        """Reject invalid input without clamping; normalize accepted numbers."""
        if self.choices:
            if not isinstance(value, str) or value not in self.choices:
                raise ValueError(f"{name} must be one of {self.choices}")
            return value
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        try:
            finite = isfinite(value)
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number")
        if self.integer and value != int(value):
            raise ValueError(f"{name} must be an integer")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{name} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} must be at most {self.maximum}")
        return int(value) if self.integer else float(value)


SETTINGS_SPECS = MappingProxyType({
    "water_depth": SettingSpec(
        "Water depth", "water", 2.0, 0.5, 4.0, "m", requires_reset=True,
    ),
    "quality": SettingSpec(
        "Particle quality", "water", "medium", choices=tuple(PARTICLE_SPACING),
        requires_reset=True,
    ),
    "wave_amplitude": SettingSpec("Wave amplitude", "waves", 0.5, 0.0, 2.0, "m"),
    "wave_period": SettingSpec("Wave period", "waves", 3.0, 0.5, 12.0, "s"),
    "wave_direction": SettingSpec(
        "Wave direction", "waves", 0.0, 0.0, 360.0, "degrees",
    ),
    "city_seed": SettingSpec(
        "City seed", "city", 0, 0, 2**32 - 1, integer=True, requires_reset=True,
    ),
    # Fraction of available procedural building lots occupied; zero means no city.
    "city_density": SettingSpec(
        "City density", "city", 0.5, 0.0, 1.0, requires_reset=True,
    ),
    # Buildings stand on the domain floor. Their top retains boundary clearance.
    "building_min_height": SettingSpec(
        "Minimum building height", "city", 1.0, 0.5, 10.0, "m", requires_reset=True,
    ),
    "building_max_height": SettingSpec(
        "Maximum building height", "city", 5.0, 0.5, 10.0, "m", requires_reset=True,
    ),
    "gravity": SettingSpec("Gravity", "physics", 9.81, 0.0, 20.0, "m/s^2"),
    "viscosity": SettingSpec(
        "Kinematic viscosity", "physics", 0.01, 0.0, 1.0, "m^2/s",
    ),
    # Exponential velocity decay per second, independent of substep frequency.
    "damping": SettingSpec("Velocity damping", "physics", 0.1, 0.0, 10.0, "1/s"),
    # Zero pauses integration; speeds above one still obey solver step limits.
    "playback_speed": SettingSpec("Playback speed", "simulation", 1.0, 0.0, 4.0, "x"),
})

RESET_FIELDS = tuple(name for name, spec in SETTINGS_SPECS.items() if spec.requires_reset)
IMMEDIATE_FIELDS = tuple(name for name, spec in SETTINGS_SPECS.items() if not spec.requires_reset)


@dataclass(frozen=True)
class Settings:
    """Validated snapshot; creating a replacement never mutates the active one."""

    water_depth: float = SETTINGS_SPECS["water_depth"].default
    quality: str = SETTINGS_SPECS["quality"].default
    wave_amplitude: float = SETTINGS_SPECS["wave_amplitude"].default
    wave_period: float = SETTINGS_SPECS["wave_period"].default
    wave_direction: float = SETTINGS_SPECS["wave_direction"].default
    city_seed: int = SETTINGS_SPECS["city_seed"].default
    city_density: float = SETTINGS_SPECS["city_density"].default
    building_min_height: float = SETTINGS_SPECS["building_min_height"].default
    building_max_height: float = SETTINGS_SPECS["building_max_height"].default
    gravity: float = SETTINGS_SPECS["gravity"].default
    viscosity: float = SETTINGS_SPECS["viscosity"].default
    damping: float = SETTINGS_SPECS["damping"].default
    playback_speed: float = SETTINGS_SPECS["playback_speed"].default

    def __post_init__(self) -> None:
        # Validate every field and cross-field constraint before normalization.
        validate_settings(self)
        for name, spec in SETTINGS_SPECS.items():
            object.__setattr__(self, name, spec.validate(name, getattr(self, name)))

    @property
    def particle_spacing(self) -> float:
        return PARTICLE_SPACING[self.quality]

    def with_updates(self, **changes: object) -> "Settings":
        """Return a validated replacement, or raise without changing this one."""
        return replace(self, **changes)

    def reset_signature(self) -> tuple[float | int | str, ...]:
        """Scene identity, excluding live gravity, viscosity, waves, and speed."""
        return tuple(getattr(self, name) for name in RESET_FIELDS)


def validate_settings(settings: Settings) -> None:
    """Raise ValueError for invalid values or incompatible scene dimensions."""
    for name, spec in SETTINGS_SPECS.items():
        spec.validate(name, getattr(settings, name))
    if settings.building_min_height > settings.building_max_height:
        raise ValueError("building_min_height must not exceed building_max_height")
    if settings.building_max_height > DOMAIN_HEIGHT - BOUNDARY_MARGIN:
        raise ValueError("building_max_height must fit inside the domain with boundary clearance")
    if settings.water_depth + settings.wave_amplitude > DOMAIN_HEIGHT - BOUNDARY_MARGIN:
        raise ValueError("water_depth plus wave_amplitude must fit inside the domain")


DEFAULT_SETTINGS = Settings()
