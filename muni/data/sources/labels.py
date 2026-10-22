"""Class-index and continuous-vector CSV data sources."""

import csv

import torch

from muni.data.core import BaseSubdataset, register_subdataset


def _load_csv_column(
    csv_path: str, column_name: str, key_format: str
) -> dict[str, str]:
    """Read one CSV column and map each data row index to a key/value pair."""
    key_to_value: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            key = key_format.format(row_idx)
            if key in key_to_value:
                raise ValueError(
                    f"key_format {key_format!r} produced duplicate key {key!r} at CSV row {row_idx}."
                )
            key_to_value[key] = row[column_name].strip()
    return key_to_value


@register_subdataset
class ClassSubdataset(BaseSubdataset):
    """Read validated class indices from a CSV column."""

    registry_name = "class"

    def __init__(
        self,
        num_classes: int,
        csv_path: str,
        column_name: str,
        class_labels: list[str] | None = None,
        key_format: str = "{}",
        drop_unknown_labels: bool = False,
    ):
        self.num_classes = int(num_classes)
        self._key_to_value: dict[str, int] = {}

        label_to_idx: dict[str, int] | None = None
        if class_labels is not None:
            if len(class_labels) != self.num_classes:
                raise ValueError(
                    f"class_labels must contain exactly num_classes={self.num_classes} entries, "
                    f"got {len(class_labels)}."
                )
            if len(set(class_labels)) != len(class_labels):
                raise ValueError(f"class_labels must be unique, got {class_labels}.")
            label_to_idx = {label: i for i, label in enumerate(class_labels)}

        raw_values = _load_csv_column(csv_path, column_name, key_format)

        for key, val in raw_values.items():
            if not val:
                continue
            if label_to_idx is not None:
                if val not in label_to_idx:
                    if drop_unknown_labels:
                        continue
                    raise ValueError(
                        f"Label '{val}' not in class_labels {class_labels}"
                    )
                v = label_to_idx[val]
            else:
                try:
                    v = int(val)
                except ValueError:
                    if drop_unknown_labels:
                        continue
                    raise
            if not (0 <= v < self.num_classes):
                if drop_unknown_labels:
                    continue
                raise ValueError(f"Label {v} not in [0, {self.num_classes})")
            self._key_to_value[key] = v

        self._keys = sorted(self._key_to_value)

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        return torch.tensor([self._key_to_value[key]], dtype=torch.long)


@register_subdataset
class SequenceSubdataset(BaseSubdataset):
    """Read fixed-length float vectors from a CSV column."""

    registry_name = "sequence"

    def __init__(
        self,
        seq_dim: int,
        csv_path: str,
        column_name: str,
        value_sep: str | None = None,
        key_format: str = "{}",
    ):
        self.seq_dim = int(seq_dim)
        self._key_to_value: dict[str, list[float]] = {}

        raw_values = _load_csv_column(csv_path, column_name, key_format)

        for key, val in raw_values.items():
            if not val:
                continue
            tokens = val.split(value_sep) if value_sep is not None else val.split()
            values = [float(x) for x in tokens if x != ""]
            if len(values) != self.seq_dim:
                raise ValueError(
                    f"Key {key}: expected {self.seq_dim} values, got {len(values)}"
                )
            self._key_to_value[key] = values

        self._keys = sorted(self._key_to_value)

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        return torch.tensor(self._key_to_value[key], dtype=torch.float32)

