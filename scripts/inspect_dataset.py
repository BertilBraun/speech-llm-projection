"""One-time inspection of the actual parquet schema and source distributions."""

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet


def main() -> None:
    table = parquet.read_table(Path("/workspace/speech-projector/data/metadata.parquet"))
    print(table.schema)
    print("ROWS", table.num_rows)
    print(
        "RANDOM_ROWS",
        table.take(np.random.default_rng(42).choice(table.num_rows, 5, replace=False)).to_pylist(),
    )
    print("SPEAKERS", Counter(table.column("speaker").to_pylist()))
    print("MODEL_DIRS", Counter(table.column("model_dir").to_pylist()))
    duration = np.array(
        [value for value in table.column("audio_duration").to_pylist() if value is not None]
    )
    lengths = np.array([len(value.split()) for value in table.column("text").to_pylist()])
    print("DURATION", np.quantile(duration, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]))
    print("WORDS", np.quantile(lengths, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]))
    print("NULLS", {name: table.column(name).null_count for name in table.column_names})


if __name__ == "__main__":
    main()
