"""Training data: one Parquet split in memory as an Arrow table, served to TensorFlow in shuffled batches.

Arrow keeps the categorical columns dictionary-encoded, so a 4-5 million row year fits in a few hundred MB.
Each epoch draws a fresh permutation and `take`s one batch at a time; only that batch is decoded to numpy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ldp.schema import CATEGORICAL, LABEL, NUMERIC


def load_split(path: str | Path, columns: list[str] | None = None) -> pa.Table:
    table = ds.dataset(str(path), format="parquet").to_table(columns=columns)
    # Dictionary-encode strings: small in memory, fast to take().
    for name in table.column_names:
        if pa.types.is_string(table.schema.field(name).type) or pa.types.is_large_string(table.schema.field(name).type):
            table = table.set_column(table.schema.get_field_index(name), name, pc.dictionary_encode(table[name]))
    return table.combine_chunks()


def to_numpy_batch(table: pa.Table) -> dict[str, np.ndarray]:
    """Model inputs for a batch: numeric float32 with NaN for null, categorical as bytes."""
    batch: dict[str, np.ndarray] = {}
    for name in NUMERIC:
        batch[name] = pc.fill_null(pc.cast(table[name], pa.float32()), np.float32("nan")).to_numpy(zero_copy_only=False)
    for name in CATEGORICAL:
        col = pc.fill_null(pc.cast(table[name], pa.string()), "missing")
        batch[name] = np.asarray(col.to_pylist(), dtype=object).astype(bytes)
    return batch


def labels(table: pa.Table) -> np.ndarray:
    return table[LABEL].to_numpy(zero_copy_only=False).astype(np.float32)


@dataclass
class BatchStream:
    table: pa.Table
    batch_size: int = 4096
    shuffle: bool = True
    seed: int = 7

    def __post_init__(self) -> None:
        self._epoch = 0

    def __len__(self) -> int:
        return (self.table.num_rows + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        n = self.table.num_rows
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            part = self.table.take(pa.array(order[start : start + self.batch_size]))
            yield to_numpy_batch(part), labels(part)


def as_dataset(stream: BatchStream):  # -> tf.data.Dataset
    import tensorflow as tf

    from ldp.training.model import input_signature

    spec = input_signature()
    ds_ = tf.data.Dataset.from_generator(
        lambda: iter(stream),
        output_signature=(spec, tf.TensorSpec(shape=(None,), dtype=tf.float32)),
    )
    # from_generator has unknown length; declaring it lets Keras end epochs without an out-of-data warning.
    return ds_.apply(tf.data.experimental.assert_cardinality(len(stream))).prefetch(tf.data.AUTOTUNE)
