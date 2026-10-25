"""Typed configuration loaded from ``configs/config.yaml``.

Every setting has a default equal to the value used in the original
notebooks, so the YAML file only needs to override what changes. Unknown
keys raise an error: a typo in the configuration must not be silently
ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    root: str = ".."
    classification_data: str = "data/raw/classification"
    denoising_train_data: str = "data/raw/classification/Photo"
    denoising_eval_data: str = "data/raw/denoising"
    coco_annotations: str = "data/raw/coco/annotations/captions_train2014.json"
    coco_images: str = "data/raw/coco/train2014"
    interim: str = "data/interim"
    artifacts: str = "artifacts"


@dataclass
class ClassifierCandidate:
    name: str
    variant: str  # reference | regularized | four_blocks
    early_stopping_start_epoch: int = 0


@dataclass
class ClassificationConfig:
    source_classes: list[str] = field(
        default_factory=lambda: ["Painting", "Photo", "Schematics", "Sketch", "Text"]
    )
    positive_class: str = "Photo"
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    known_invalid_files: list[str] = field(default_factory=list)
    image_size: tuple[int, int] = (180, 180)
    data_fraction: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    max_epochs: int = 40
    learning_rate: float = 3e-4
    clipnorm: float = 1.0
    cache_workers: int = 8
    candidates: list[ClassifierCandidate] = field(
        default_factory=lambda: [
            ClassifierCandidate("cnn_reference", "reference"),
            ClassifierCandidate("cnn_regularized", "regularized"),
            ClassifierCandidate("cnn_4_blocks", "four_blocks", 15),
        ]
    )
    enriched_augmentation: bool = False
    threshold_grid: dict[str, float] = field(default_factory=lambda: {"start": 0.10, "stop": 0.90, "num": 81})
    reduce_lr: dict[str, float] = field(
        default_factory=lambda: {"factor": 0.5, "patience": 2, "min_delta": 0.001, "min_lr": 1e-6}
    )
    early_stopping: dict[str, float] = field(default_factory=lambda: {"patience": 5, "min_delta": 0.001})

    def __post_init__(self) -> None:
        self.image_size = tuple(self.image_size)  # type: ignore[assignment]
        if self.positive_class not in self.source_classes:
            raise ConfigError("classification.positive_class must be one of source_classes")
        if not 0 < self.data_fraction <= 1:
            raise ConfigError("classification.data_fraction must be in (0, 1]")
        if self.val_fraction <= 0 or self.test_fraction <= 0:
            raise ConfigError("classification val/test fractions must be positive")
        if self.val_fraction + self.test_fraction >= 1:
            raise ConfigError("classification val_fraction + test_fraction must be < 1")
        if not self.candidates:
            raise ConfigError("classification.candidates must not be empty")


@dataclass
class DenoisingConfig:
    image_size: int = 128
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    max_train_images: int = 4000
    batch_size: int = 32
    max_epochs: int = 25
    early_stopping_patience: int = 4
    duplicate_threshold: float = 0.02
    noise_levels: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "gaussian": {"low": 0.05, "medium": 0.10, "high": 0.20},
            "salt_pepper": {"low": 0.02, "medium": 0.05, "high": 0.10},
            "speckle": {"low": 0.10, "medium": 0.25, "high": 0.50},
        }
    )
    val_noise_seed: int = 1000
    test_noise_seed: int = 2000
    architecture: str = "A2_unet"
    loss: str = "mse"
    learning_rate: float = 0.003
    train_noise: str = "mixed"
    compare_architectures: list[str] = field(
        default_factory=lambda: ["A0_base", "A1_wide", "A2_unet", "A3_strided"]
    )

    def __post_init__(self) -> None:
        if self.image_size % 4:
            raise ConfigError("denoising.image_size must be a multiple of 4")
        for name, levels in self.noise_levels.items():
            if set(levels) != {"low", "medium", "high"}:
                raise ConfigError(f"denoising.noise_levels.{name} needs low/medium/high")
        if self.train_noise != "mixed" and self.train_noise not in self.noise_levels:
            raise ConfigError("denoising.train_noise must be 'mixed' or a configured noise")


@dataclass
class CaptionModelConfig:
    cnn: str = "InceptionResNetV2"
    rnn: str = "lstm"
    initial_state: bool = True
    units: int = 512
    embedding_dim: int = 256
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.rnn not in {"gru", "lstm"}:
            raise ConfigError("captioning.model.rnn must be 'gru' or 'lstm'")
        if not 0 <= self.dropout < 1:
            raise ConfigError("captioning.model.dropout must be in [0, 1)")


@dataclass
class CaptioningConfig:
    n_test: int = 1000
    n_val: int = 1000
    n_train: int = 40000
    vocab_size: int = 5000
    max_len_percentile: float = 99
    feature_batch_size: int = 16
    max_epochs: int = 30
    patience: int = 3
    decoding: str = "greedy"
    generation_batch_size: int = 250
    model: CaptionModelConfig = field(default_factory=CaptionModelConfig)

    def __post_init__(self) -> None:
        if self.decoding not in {"greedy", "sample"}:
            raise ConfigError("captioning.decoding must be 'greedy' or 'sample'")


@dataclass
class InferenceConfig:
    classifier_run: str | None = None
    denoiser_run: str | None = None
    captioner_run: str | None = None
    max_denoise_side: int | None = None


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # Directory the relative paths are resolved against (set by load_config).
    base_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH.parent)

    # -- path helpers -------------------------------------------------------
    @property
    def root(self) -> Path:
        return (self.base_dir / self.paths.root).resolve()

    def path(self, name: str) -> Path:
        """Absolute path for one of the entries of ``paths``."""
        value = getattr(self.paths, name)
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def resolve(self, value: str | Path) -> Path:
        """Resolve a user-supplied path (absolute, or relative to the repository root)."""
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("base_dir")
        return data


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _build(cls: type[T], data: dict[str, Any] | None, where: str) -> T:
    """Instantiate a (nested) dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    fields = {f.name: f for f in dataclasses.fields(cls) if f.name != "base_dir"}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        nested = _NESTED.get((cls.__name__, name))
        if nested is not None and isinstance(value, dict):
            kwargs[name] = _build(nested, value, f"{where}.{name}")
        elif (cls.__name__, name) == ("ClassificationConfig", "candidates"):
            kwargs[name] = [_build(ClassifierCandidate, c, f"{where}.candidates") for c in value]
        else:
            # Value constraints are checked in each dataclass's __post_init__.
            kwargs[name] = value
    return cls(**kwargs)


_NESTED: dict[tuple[str, str], type] = {
    ("Config", "paths"): PathsConfig,
    ("Config", "classification"): ClassificationConfig,
    ("Config", "denoising"): DenoisingConfig,
    ("Config", "captioning"): CaptioningConfig,
    ("Config", "inference"): InferenceConfig,
    ("CaptioningConfig", "model"): CaptionModelConfig,
}


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration (defaults to ``configs/config.yaml``)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")
    config = _build(Config, raw, "config")
    config.base_dir = config_path.parent
    return config


def config_from_dict(data: dict[str, Any], base_dir: str | Path) -> Config:
    """Build a configuration from a dict (used by tests)."""
    config = _build(Config, data, "config")
    config.base_dir = Path(base_dir).resolve()
    return config
