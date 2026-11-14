"""Strict, file-relative training configuration for Experiment 04."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib

from forward_dynamics.serialization import require_fields, sha256_file
from learned_dynamics.checkpoint_metadata import REPOSITORY_ROOT
from learned_dynamics.types import DATASET_VERSION, ModelSpec


PROTOCOL_VARIANTS = {
    "canonical32": (32, 0, "one_step", "all"),
    "capacity16": (16, 0, "one_step", "all"),
    "history0": (32, 0, "one_step", "history_common"),
    "history1": (32, 1, "one_step", "history_common"),
    "loss1": (32, 0, "one_step", "loss_common"),
    "loss5": (32, 0, "rollout_5", "loss_common"),
}
PROTOCOL_BATCH_SIZE = 64
PROTOCOL_SAMPLER_SEED_OFFSET = 100000
PROTOCOL_VALIDATION_INTERVAL_EPOCHS = 1
PROTOCOL_ARTIFACT_BUDGET_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float

    def __post_init__(self) -> None:
        values = (self.learning_rate, *self.betas, self.epsilon, self.weight_decay)
        if self.name != "adam" or any(type(value) not in (int, float) or not math.isfinite(value)
                                      for value in values):
            raise ValueError("optimizer values must be finite Adam settings")
        if (self.learning_rate <= 0 or self.betas != (0.9, 0.999) or
                self.epsilon != 1e-8 or self.weight_decay != 0.0):
            raise ValueError("optimizer settings do not match protocol v1")


@dataclass(frozen=True)
class RunSpec:
    variant_id: str
    model: ModelSpec
    loss: str
    membership: str
    seed: int
    device: str
    repeat: int

    def __post_init__(self) -> None:
        expected = PROTOCOL_VARIANTS.get(self.variant_id)
        actual = (self.model.hidden_width, self.model.history_steps, self.loss, self.membership)
        if expected is None or actual != expected:
            raise ValueError("run variant/model/loss/membership does not match protocol")
        if type(self.seed) is not int or self.seed not in range(5):
            raise ValueError("seed must be a declared integer in 0..4")
        if self.device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        if type(self.repeat) is not int or self.repeat not in (0, 1):
            raise ValueError("repeat must be a declared integer in 0..1")

    @property
    def run_id(self) -> str:
        return f"{self.variant_id}-{self.device}-seed{self.seed}-repeat{self.repeat}"

    def mapping(self) -> dict:
        return {
            "variant_id": self.variant_id, "model": asdict(self.model),
            "loss": self.loss, "membership": self.membership, "seed": self.seed,
            "device": self.device, "repeat": self.repeat,
        }


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: int
    config_path: Path
    config_sha256: str
    dataset_dir: Path
    dataset_version: str
    output_dir: Path
    runs: tuple[RunSpec, ...]
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    sampler_seed_offset: int
    validation_interval_epochs: int
    selection_metric: str
    tie_break: str
    normalization_statistics: str
    constant_threshold: float
    target_transform: str
    artifact_budget_bytes: int
    resume: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported training config schema")
        for field in ("epochs", "batch_size", "sampler_seed_offset",
                      "validation_interval_epochs", "artifact_budget_bytes"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not 1 <= self.epochs <= 300:
            raise ValueError("epochs must be in 1..300")
        if (self.batch_size != PROTOCOL_BATCH_SIZE or
                self.sampler_seed_offset != PROTOCOL_SAMPLER_SEED_OFFSET or
                self.validation_interval_epochs != PROTOCOL_VALIDATION_INTERVAL_EPOCHS or
                self.artifact_budget_bytes != PROTOCOL_ARTIFACT_BUDGET_BYTES):
            raise ValueError("training settings do not match protocol v1")
        if self.dataset_version != DATASET_VERSION:
            raise ValueError("unsupported dataset version")
        if (self.selection_metric != "validation_normalized_one_step_mse" or
                self.tie_break != "earliest_epoch"):
            raise ValueError("unsupported checkpoint selection policy")
        if (self.normalization_statistics != "float64_population_std" or
                type(self.constant_threshold) is not float or
                self.constant_threshold != 1e-12 or self.target_transform != "next_minus_current"):
            raise ValueError("normalization settings do not match protocol v1")
        if self.resume != "reject":
            raise ValueError("resume is unsupported; restart from scratch")
        if not isinstance(self.runs, tuple) or not self.runs:
            raise ValueError("runs must be a nonempty immutable roster")
        ids = [run.run_id for run in self.runs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate run id in training roster")


TOP_FIELDS = {
    "schema_version", "dataset_dir", "dataset_version", "output_dir",
    "artifact_budget_bytes", "resume", "training", "optimizer", "normalization", "runs",
}
TRAINING_FIELDS = {
    "epochs", "batch_size", "sampler_seed_offset", "validation_interval_epochs",
    "selection_metric", "tie_break",
}
OPTIMIZER_FIELDS = {"name", "learning_rate", "beta1", "beta2", "epsilon", "weight_decay"}
NORMALIZATION_FIELDS = {"statistics", "constant_threshold", "target_transform"}
RUN_FIELDS = {
    "variant_id", "hidden_width", "history_steps", "loss", "membership", "seed", "device", "repeat",
}


def _relative_path(config_path: Path, raw: object, description: str) -> Path:
    if (not isinstance(raw, str) or not raw or raw.startswith("~") or "$" in raw or
            Path(raw).is_absolute()):
        raise ValueError(f"{description} must be a literal relative path")
    return (config_path.parent / raw).resolve()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validate_output(config_path: Path, dataset: Path, output: Path) -> None:
    experiment = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    protected = [dataset, *(experiment / name for name in ("src", "scripts", "tests"))]
    config_overlap = (output == config_path.parent or config_path.is_relative_to(output) or
                      (config_path.parent.name == "configs" and output.is_relative_to(config_path.parent)))
    if config_overlap or any(_overlaps(output, path) for path in protected if path.exists()):
        raise ValueError("training output cannot overlap dataset/config/source inputs")
    if output.is_relative_to(REPOSITORY_ROOT) and not (
        output.is_relative_to(REPOSITORY_ROOT / "artifacts") or
        output.is_relative_to(REPOSITORY_ROOT / "experiments/04-learned-dynamics/results")
    ):
        raise ValueError("training output overlaps protected repository source")


def load_training_config(path: Path) -> TrainingConfig:
    """Load exact TOML fields; paths are literal and relative to the file."""
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot read training config") from error
    require_fields(raw, TOP_FIELDS, "training config")
    require_fields(raw["training"], TRAINING_FIELDS, "training settings")
    require_fields(raw["optimizer"], OPTIMIZER_FIELDS, "optimizer settings")
    require_fields(raw["normalization"], NORMALIZATION_FIELDS, "normalization settings")
    if not isinstance(raw["runs"], list) or not raw["runs"]:
        raise ValueError("runs must be a nonempty TOML array of tables")
    dataset = _relative_path(config_path, raw["dataset_dir"], "dataset_dir")
    if not dataset.is_dir():
        raise ValueError("dataset_dir must identify an existing directory")
    output = _relative_path(config_path, raw["output_dir"], "output_dir")
    _validate_output(config_path, dataset, output)
    runs = []
    for value in raw["runs"]:
        require_fields(value, RUN_FIELDS, "run specification")
        runs.append(RunSpec(
            variant_id=value["variant_id"],
            model=ModelSpec(value["hidden_width"], value["history_steps"]),
            loss=value["loss"], membership=value["membership"], seed=value["seed"],
            device=value["device"], repeat=value["repeat"],
        ))
    training, optimizer, normalization = raw["training"], raw["optimizer"], raw["normalization"]
    return TrainingConfig(
        schema_version=raw["schema_version"], config_path=config_path,
        config_sha256=sha256_file(config_path), dataset_dir=dataset,
        dataset_version=raw["dataset_version"], output_dir=output, runs=tuple(runs),
        epochs=training["epochs"], batch_size=training["batch_size"],
        optimizer=OptimizerConfig(optimizer["name"], optimizer["learning_rate"],
                                  (optimizer["beta1"], optimizer["beta2"]),
                                  optimizer["epsilon"], optimizer["weight_decay"]),
        sampler_seed_offset=training["sampler_seed_offset"],
        validation_interval_epochs=training["validation_interval_epochs"],
        selection_metric=training["selection_metric"], tie_break=training["tie_break"],
        normalization_statistics=normalization["statistics"],
        constant_threshold=normalization["constant_threshold"],
        target_transform=normalization["target_transform"],
        artifact_budget_bytes=raw["artifact_budget_bytes"], resume=raw["resume"],
    )
