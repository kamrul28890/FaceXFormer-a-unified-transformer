"""
Shared helpers for FaceXFormer baseline and ablation runs.

These utilities support both small local subsets and full available evaluation
splits. The ablation runner should usually stay tiny on local hardware; the
baseline runner can be used for either smoke tests or current 8-task eval.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


TASK_IDS = {
    "segmentation": 0,
    "landmark": 1,
    "headpose": 2,
    "attribute": 3,
    "age": 4,
    "gender": 5,
    "race": 6,
    "visibility": 7,
}


PAPER_TARGETS = {
    "segmentation": {"dataset": "CelebAMask-HQ", "metric": "F1", "target": 92.01, "higher_is_better": True},
    "landmark": {"dataset": "300W", "metric": "NME", "target": 4.67, "higher_is_better": False},
    "headpose": {"dataset": "BIWI", "metric": "MAE", "target": 3.52, "higher_is_better": False},
    "attribute": {"dataset": "CelebA", "metric": "Accuracy", "target": 91.83, "higher_is_better": True},
    "age": {"dataset": "UTKFace", "metric": "MAE", "target": 4.17, "higher_is_better": False},
    "visibility": {"dataset": "COFW", "metric": "Recall@P80", "target": 72.56, "higher_is_better": True},
}


class NamedSubset(Dataset):
    """Subset wrapper that keeps a readable dataset name in reports."""

    def __init__(self, dataset: Dataset, max_samples: int, seed: int = 0, name: Optional[str] = None):
        self.dataset = dataset
        n = len(dataset)
        if max_samples <= 0 or max_samples >= n:
            indices = list(range(n))
            self.selection_label = "all"
        else:
            rng = random.Random(seed)
            indices = list(range(n))
            rng.shuffle(indices)
            indices = sorted(indices[:max_samples])
            self.selection_label = "subset"
        self.subset = Subset(dataset, indices)
        self.name = name or dataset_name(dataset)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        return self.subset[idx]

    def get_name(self) -> str:
        return f"{self.name}[{self.selection_label}:{len(self)}]"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy/scalar values produced by metrics."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_root(user_root: str) -> Path:
    """
    Resolve the dataset root used by the current copied dataset layout.

    The repo currently has both `datasets/` and a nested `datasets/datasets/`.
    The nested folder appears to contain the most complete copy, including
    300W and 300W-LP, so we prefer it when present.
    """

    root = Path(user_root).resolve()
    nested = root / "datasets"
    if nested.exists() and (nested / "300w").exists():
        return nested
    return root


def dataset_name(dataset: Dataset) -> str:
    if hasattr(dataset, "get_name"):
        return dataset.get_name()
    return dataset.__class__.__name__


def limit_dataset(dataset: Dataset, max_samples: int, seed: int) -> NamedSubset:
    return NamedSubset(dataset, max_samples=max_samples, seed=seed)


def _try_make(factory, task_name: str, missing: List[str]):
    try:
        return factory()
    except Exception as exc:
        missing.append(f"{task_name}: {exc}")
        return None


def build_eval_datasets(dataset_root: Path, tasks: Iterable[str], max_samples: int, seed: int):
    """
    Build evaluation datasets for the current 8-task implementation.

    Expression and face recognition are intentionally not included here because
    the current repo does not have those model heads/losses wired yet.
    """

    from datasets import (
        BIWIDataset,
        COFWDataset,
        CelebADataset,
        CelebAMaskHQDataset,
        FairFaceDataset,
        LFWADataset,
        MultiLabelDatasetWrapper,
        UTKFaceDataset,
        W300Dataset,
    )

    root = str(dataset_root)
    selected = set(tasks)
    datasets: Dict[str, List[Dataset]] = {}
    missing: List[str] = []

    if "segmentation" in selected:
        ds = _try_make(lambda: CelebAMaskHQDataset("test", dataset_root=root), "segmentation", missing)
        if ds is not None:
            datasets["segmentation"] = [limit_dataset(ds, max_samples, seed)]

    if "landmark" in selected:
        full = _try_make(lambda: W300Dataset("test_full", dataset_root=root), "landmark/300W-full", missing)
        common = _try_make(lambda: W300Dataset("test_common", dataset_root=root), "landmark/300W-common", missing)
        challenging = _try_make(lambda: W300Dataset("test_challenging", dataset_root=root), "landmark/300W-challenging", missing)
        landmark_sets = [ds for ds in [full, common, challenging] if ds is not None]
        if landmark_sets:
            datasets["landmark"] = [
                limit_dataset(ds, max_samples, seed + offset)
                for offset, ds in enumerate(landmark_sets)
            ]

    if "headpose" in selected:
        ds = _try_make(lambda: BIWIDataset("test", dataset_root=root), "headpose", missing)
        if ds is not None:
            datasets["headpose"] = [limit_dataset(ds, max_samples, seed)]

    if "attribute" in selected:
        ds = _try_make(lambda: CelebADataset("test", dataset_root=root), "attribute", missing)
        if ds is not None:
            datasets["attribute"] = [limit_dataset(ds, max_samples, seed)]
        lfwa = _try_make(lambda: LFWADataset("test", dataset_root=root), "attribute/LFWA", missing)
        if lfwa is not None:
            datasets.setdefault("attribute", []).append(limit_dataset(lfwa, max_samples, seed + 1))

    if {"age", "gender", "race"} & selected:
        utk = _try_make(lambda: UTKFaceDataset("test", dataset_root=root), "UTKFace", missing)
        fair = _try_make(lambda: FairFaceDataset("test", dataset_root=root), "FairFace", missing)
        bases = [ds for ds in [utk, fair] if ds is not None]
        for task in ["age", "gender", "race"]:
            if task in selected:
                wrapped = [MultiLabelDatasetWrapper(base, task) for base in bases]
                if wrapped:
                    datasets[task] = [limit_dataset(ds, max_samples, seed) for ds in wrapped]

    if "visibility" in selected:
        ds = _try_make(lambda: COFWDataset("test", dataset_root=root), "visibility", missing)
        if ds is not None:
            datasets["visibility"] = [limit_dataset(ds, max_samples, seed)]

    return datasets, missing


def build_train_datasets(dataset_root: Path, tasks: Iterable[str], max_samples: int, seed: int):
    """Build tiny train datasets for local ablation smoke runs."""

    from datasets import (
        COFWDataset,
        CelebADataset,
        CelebAMaskHQDataset,
        FairFaceDataset,
        MultiLabelDatasetWrapper,
        UTKFaceDataset,
        W300Dataset,
        W300LPDataset,
    )

    root = str(dataset_root)
    selected = set(tasks)
    datasets: Dict[str, List[Dataset]] = {}
    missing: List[str] = []

    if "segmentation" in selected:
        ds = _try_make(lambda: CelebAMaskHQDataset("train", dataset_root=root), "segmentation", missing)
        if ds is not None:
            datasets["segmentation"] = [limit_dataset(ds, max_samples, seed)]

    if "landmark" in selected:
        ds = _try_make(lambda: W300Dataset("train", dataset_root=root), "landmark", missing)
        if ds is not None:
            datasets["landmark"] = [limit_dataset(ds, max_samples, seed)]

    if "headpose" in selected:
        ds = _try_make(lambda: W300LPDataset("train", dataset_root=root), "headpose", missing)
        if ds is not None:
            datasets["headpose"] = [limit_dataset(ds, max_samples, seed)]

    if "attribute" in selected:
        ds = _try_make(lambda: CelebADataset("train", dataset_root=root), "attribute", missing)
        if ds is not None:
            datasets["attribute"] = [limit_dataset(ds, max_samples, seed)]

    if {"age", "gender", "race"} & selected:
        utk = _try_make(lambda: UTKFaceDataset("train", dataset_root=root), "UTKFace", missing)
        fair = _try_make(lambda: FairFaceDataset("train", dataset_root=root), "FairFace", missing)
        bases = [ds for ds in [utk, fair] if ds is not None]
        for task in ["age", "gender", "race"]:
            if task in selected:
                wrapped = [MultiLabelDatasetWrapper(base, task) for base in bases]
                if wrapped:
                    datasets[task] = [limit_dataset(ds, max_samples, seed) for ds in wrapped]

    if "visibility" in selected:
        ds = _try_make(lambda: COFWDataset("train", dataset_root=root), "visibility", missing)
        if ds is not None:
            datasets["visibility"] = [limit_dataset(ds, max_samples, seed)]

    return datasets, missing


def load_checkpoint_if_available(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
    strict: bool = True,
) -> bool:
    """
    Load a checkpoint if supplied.

    Returns True when weights were loaded. If no checkpoint is provided, the
    caller can still run a random-weight smoke test.
    """

    if not checkpoint_path:
        return False

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Supported checkpoint layouts:
    # - our training code: {"model_state_dict": ...}
    # - released inference code in this repo: {"state_dict_backbone": ...}
    # - common PyTorch exports: {"state_dict": ...}
    # - raw state_dict: {"layer.weight": tensor, ...}
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict_backbone")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_state_dict[key.replace("module.", "")] = value

    if strict:
        model.load_state_dict(clean_state_dict, strict=True)
    else:
        # Public inference checkpoints and local training checkpoints sometimes
        # differ in prefixes or task-head names. For local smoke testing we can
        # load only shape-compatible tensors and report how much matched.
        model_state = model.state_dict()
        compatible = OrderedDict()
        skipped = []
        for key, value in clean_state_dict.items():
            if hasattr(value, "shape") and key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)

        model_state.update(compatible)
        model.load_state_dict(model_state, strict=True)
        print(f"Loaded {len(compatible)} checkpoint tensors; skipped {len(skipped)} incompatible tensors.")
    return True


def predictions_from_outputs(outputs: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
    """Convert the current FaceXFormer tuple output into the loss/metric dict."""

    (
        landmark_out,
        headpose_out,
        attribute_out,
        visibility_out,
        age_out,
        gender_out,
        race_out,
        seg_out,
    ) = outputs

    return {
        "landmark_output": landmark_out,
        "headpose_output": headpose_out,
        "attribute_output": attribute_out,
        "visibility_output": visibility_out,
        "age_output": age_out,
        "gender_output": gender_out,
        "race_output": race_out,
        "seg_output": seg_out,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
