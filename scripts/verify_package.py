"""Verify a BioImage.IO package the way a consumer would, without our source tree.

`bioimageio test` is the authority, but this checks the parts that fail silently:
the state dict reloads from the packaged file *and* the packaged architecture, the
packaged test output is really what those weights produce, and every sha256 in
SHA256SUMS still matches. A package that fails any of these is not shippable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_architecture(path: Path):
    """Import the packaged architecture file directly, ignoring anything installed."""
    spec = importlib.util.spec_from_file_location("packaged_unet", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["packaged_unet"] = module
    spec.loader.exec_module(module)
    return module.UNet3D


def main() -> None:
    root = Path(sys.argv[1])
    rdf = yaml.safe_load((root / "rdf.yaml").read_text())
    weights_entry = rdf["weights"]["pytorch_state_dict"]
    failures: list[str] = []

    # 1. checksums
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256(root / name) != digest:
            failures.append(f"checksum mismatch: {name}")
    print(f"checked {(root / 'SHA256SUMS').read_text().count(chr(10))} checksums")

    # 2. rdf hashes agree with the files
    for key in ("sha256",):
        declared = weights_entry[key]
        if declared != sha256(root / weights_entry["source"]):
            failures.append("rdf weights sha256 does not match weights.pt")
    architecture = weights_entry["architecture"]
    if architecture["sha256"] != sha256(root / architecture["source"]):
        failures.append("rdf architecture sha256 does not match the packaged file")

    # 3. architecture loads and the state dict fits strictly
    UNet3D = load_architecture(root / architecture["source"])
    model = UNet3D(**architecture["kwargs"])
    state = torch.load(root / weights_entry["source"], map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    model.eval()
    print(f"reloaded {len(model.state_dict())} tensors strictly via the packaged architecture")

    # 4. the packaged test output is what this model actually produces
    sample_input = np.load(root / rdf["inputs"][0]["test_tensor"]["source"])
    expected = np.load(root / rdf["outputs"][0]["test_tensor"]["source"])
    with torch.no_grad():
        actual = model(torch.from_numpy(sample_input).float()).numpy()
    max_error = float(np.abs(actual - expected).max())
    print(f"input {sample_input.shape} -> output {actual.shape}, max error vs packaged {max_error:.2e}")
    if actual.shape != expected.shape:
        failures.append(f"output shape {actual.shape} != packaged {expected.shape}")
    if max_error > 1e-5:
        failures.append(f"reproduced output differs by {max_error:.2e}")

    # 5. every declared axis is satisfiable by the packaged tensors
    for entry in rdf["inputs"] + rdf["outputs"]:
        tensor = np.load(root / entry["test_tensor"]["source"])
        declared = len(entry["axes"])
        if tensor.ndim != declared:
            failures.append(
                f"{entry['id']}: tensor has {tensor.ndim} dims but {declared} axes are declared"
            )
    print(f"axis declarations match packaged tensor ranks")

    # 6. the model must refuse an illegal shape, proving the multiple-of-4 contract is real
    illegal = torch.zeros(1, 1, 30, 128, 128)
    try:
        with torch.no_grad():
            model(illegal)
    except RuntimeError:
        print("illegal 30-voxel axis rejected, as documented")
    else:
        failures.append("model accepted a 30-voxel axis; the documented constraint is wrong")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\npackage verified")


if __name__ == "__main__":
    main()
