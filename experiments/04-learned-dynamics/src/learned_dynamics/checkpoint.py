"""Inference-only checkpoint bundles bound to original train data and scaling."""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from forward_dynamics.serialization import decode_object, json_bytes, require_equal, require_fields, sha256_file
from learned_dynamics.checkpoint_metadata import (
    DATASET_DIR, DERIVED_FIELDS, METADATA_FIELDS, REPOSITORY_ROOT, validate_metadata,
)
from learned_dynamics.checkpoint_tensors import MAX_TENSOR_BYTES, load_tensors
from learned_dynamics.data import load_views
from learned_dynamics.devices import require_device
from learned_dynamics.model import ResidualMLP
from learned_dynamics.normalization import fit_normalizer, load_normalizer, write_normalizer
from learned_dynamics.types import Normalizer


MAX_JSON_BYTES = 2 * 1024 * 1024
FILES = ("metadata.json", "normalizer.json", "weights.npz")


@dataclass(frozen=True)
class CheckpointBundle:
    model: ResidualMLP
    normalizer: Normalizer
    metadata: dict
    bundle_sha256: str


def _read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("checkpoint JSON exceeds size limit")
    return decode_object(path.read_text())


def load_checkpoint(input_dir: Path, device: str = "cpu") -> CheckpointBundle:
    """Audit against the committed v1 dataset before materializing a model.

    Config/source/seed fields are declared provenance, not proof of training.
    Config/selection replay and source-tree audit belong to later training/evidence work.
    """
    directory = Path(input_dir)
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("checkpoint must be a real local directory")
        if {path.name for path in directory.iterdir()} != set(FILES) | {"manifest.json"}:
            raise ValueError("checkpoint files do not match the fixed bundle layout")
        if any((directory / name).is_symlink() or not (directory / name).is_file()
               for name in (*FILES, "manifest.json")):
            raise ValueError("checkpoint files must be regular files, not symlinks")
        for name in (*FILES, "manifest.json"):
            limit = MAX_TENSOR_BYTES if name == "weights.npz" else MAX_JSON_BYTES
            if (directory / name).stat().st_size > limit:
                raise ValueError("checkpoint file exceeds size limit")
        manifest = _read_json(directory / "manifest.json")
        require_equal(manifest, {"schema_version": 1, "files": {
            name: sha256_file(directory / name) for name in FILES}}, "checkpoint manifest")
        normalizer = load_normalizer(directory / "normalizer.json")
        # Independently refit and compare identities/statistics, rather than
        # treating repaired outer hashes as evidence of a genuine train fit.
        expected = fit_normalizer(load_views(DATASET_DIR, normalizer.dataset_version).train)
        require_equal(asdict(normalizer), asdict(expected), "normalizer source fit")
        metadata = _read_json(directory / "metadata.json")
        require_equal(metadata.get("normalizer_sha256"), sha256_file(directory / "normalizer.json"), "normalizer hash")
        require_equal(metadata.get("tensor_sha256"), sha256_file(directory / "weights.npz"), "tensor hash")
        spec = validate_metadata(metadata, normalizer)
        tensors = load_tensors(directory / "weights.npz", spec)
        target = require_device(device)
        model = ResidualMLP(spec, seed=metadata["run_spec"]["seed"])
        model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in tensors.items()}, strict=True)
        model.to(target).eval()
        return CheckpointBundle(model, normalizer, metadata, sha256_file(directory / "manifest.json"))
    except (OSError, UnicodeError, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"cannot load checkpoint: {error}") from error


def save_checkpoint(output_dir: Path, model: ResidualMLP, normalizer: Normalizer,
                    metadata: dict) -> Path:
    """Save metadata without derived hash fields; validate staged bundle before copying.

    Publication reserves the destination exclusively. A crash during copying can
    leave an incomplete directory; loaders reject it and a new output path is needed.
    """
    output = Path(output_dir)
    if output.is_symlink() or output.exists():
        raise ValueError("checkpoint output already exists or is a symlink")
    output = output.resolve()
    if output.is_relative_to(REPOSITORY_ROOT) and not (
        output.is_relative_to(REPOSITORY_ROOT / "artifacts") or
        output.is_relative_to(REPOSITORY_ROOT / "experiments/04-learned-dynamics/results")
    ):
        raise ValueError("checkpoint output overlaps protected source/input paths")
    if not isinstance(model, ResidualMLP) or not isinstance(normalizer, Normalizer):
        raise ValueError("checkpoint requires a model and normalizer")
    require_fields(metadata, METADATA_FIELDS - DERIVED_FIELDS, "checkpoint input metadata")
    require_equal(metadata["model_spec"], asdict(model.spec), "saved model architecture")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-stage-", dir=output.parent) as temp:
        stage = Path(temp)
        write_normalizer(normalizer, stage / "normalizer.json")
        np.savez(stage / "weights.npz", **{
            key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        payload = dict(metadata, normalizer_sha256=sha256_file(stage / "normalizer.json"),
                       tensor_file="weights.npz", tensor_sha256=sha256_file(stage / "weights.npz"))
        (stage / "metadata.json").write_bytes(json_bytes(payload, indent=2))
        (stage / "manifest.json").write_bytes(json_bytes({
            "schema_version": 1, "files": {name: sha256_file(stage / name) for name in FILES}}, indent=2))
        load_checkpoint(stage, device="cpu")
        try:
            output.mkdir()
        except FileExistsError as error:
            raise ValueError("checkpoint output already exists") from error
        for name in (*FILES, "manifest.json"):
            shutil.copyfile(stage / name, output / name)
    return output
