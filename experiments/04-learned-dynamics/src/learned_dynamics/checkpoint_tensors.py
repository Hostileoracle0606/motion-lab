"""Bounded NPZ audit before NumPy or PyTorch can allocate tensor storage."""

from pathlib import Path
import zipfile

import numpy as np

from learned_dynamics.types import ModelSpec


MAX_TENSOR_BYTES = 16 * 1024 * 1024


def expected_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    width = spec.hidden_width
    return {"network.0.weight": (width, 3 * (1 + spec.history_steps)),
            "network.0.bias": (width,), "network.2.weight": (width, width),
            "network.2.bias": (width,), "network.4.weight": (2, width),
            "network.4.bias": (2,)}


def load_tensors(path: Path, spec: ModelSpec) -> dict[str, np.ndarray]:
    shapes = expected_shapes(spec)
    if path.stat().st_size > MAX_TENSOR_BYTES:
        raise ValueError("checkpoint tensor archive exceeds size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(shapes) or set(names) != {key + ".npy" for key in shapes}:
                raise ValueError("tensor archive members do not match expected names")
            if sum(entry.file_size for entry in entries) > MAX_TENSOR_BYTES:
                raise ValueError("decompressed tensor archive exceeds size limit")
            for entry in entries:
                if entry.flag_bits & 1:
                    raise ValueError("encrypted tensor archive is unsupported")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("unsupported NPY header version")
                    if shape != shapes[entry.filename[:-4]] or dtype != np.dtype("float32") or fortran:
                        raise ValueError("tensor header has wrong shape/dtype/order")
        with np.load(path, allow_pickle=False) as archive:
            tensors = {key: archive[key] for key in shapes}
        if any(not np.isfinite(value).all() for value in tensors.values()):
            raise ValueError("checkpoint tensors must be finite")
        return tensors
    except (OSError, EOFError, zipfile.BadZipFile, ValueError) as error:
        raise ValueError(f"invalid checkpoint tensors: {error}") from error
