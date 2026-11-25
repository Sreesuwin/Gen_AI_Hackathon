"""Build the complete M1 processed Parquet datasets from the Ausgrid CSVs.

Reuses ``src.pipeline.load`` utilities and writes:
- ``data/processed/halfhourly.parquet`` (Section 1.1 schema)
- ``data/processed/homes.parquet`` (Section 1.2 schema)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.load import (
    HALFHOURLY_COLUMNS,
    _get_processed_dir,
    reshape_ausgrid_csv,
    resolve_dataset_dir,
)

AUSGRID_CSV_FILES = (
    "Solar home 2010-2011.csv",
    "Solar home 2011-2012.csv",
    "Solar home 2012-2013.csv",
)

HOMES_COLUMNS = ["home_id", "pv_kw", "postcode", "has_cl", "n_days"]


def _extract_csv_metadata(csv_path: Path) -> pd.DataFrame:
    """Extract unique (home_id, pv_kw, postcode) rows from one Ausgrid CSV."""
    meta = pd.read_csv(
        csv_path,
        skiprows=1,
        usecols=["Customer", "Generator Capacity", "Postcode"],
    )
    meta = meta.drop_duplicates(subset=["Customer"]).rename(
        columns={
            "Customer": "home_id",
            "Generator Capacity": "pv_kw",
            "Postcode": "postcode",
        }
    )
    return meta


def build_all(
    dataset_dir: Path | None = None,
    processed_dir: Path | None = None,
    row_group_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all three Ausgrid CSVs into ``halfhourly.parquet`` and ``homes.parquet``."""
    src_dir = Path(dataset_dir) if dataset_dir is not None else resolve_dataset_dir()
    out_dir = Path(processed_dir) if processed_dir is not None else _get_processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    yearly_frames: list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []

    for filename in AUSGRID_CSV_FILES:
        csv_path = src_dir / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Expected Ausgrid CSV not found: {csv_path}")
        yearly_frames.append(reshape_ausgrid_csv(csv_path))
        meta_frames.append(_extract_csv_metadata(csv_path))

    halfhourly = (
        pd.concat(yearly_frames, ignore_index=True)[HALFHOURLY_COLUMNS]
        .sort_values(["home_id", "ts"])
        .reset_index(drop=True)
    )

    base_meta = (
        pd.concat(meta_frames, ignore_index=True)
        .drop_duplicates(subset=["home_id"])
        .sort_values("home_id")
    )
    home_stats = (
        halfhourly.assign(_day=halfhourly["ts"].dt.normalize())
        .groupby("home_id", as_index=False)
        .agg(
            has_cl=("cl_kwh", lambda s: bool((s > 0).any())),
            n_days=("_day", "nunique"),
        )
    )
    homes = base_meta.merge(home_stats, on="home_id", how="inner")
    homes["home_id"] = homes["home_id"].astype("int64")
    homes["pv_kw"] = homes["pv_kw"].astype("float64")
    homes["postcode"] = homes["postcode"].astype("int64")
    homes["has_cl"] = homes["has_cl"].astype("bool")
    homes["n_days"] = homes["n_days"].astype("int64")
    homes = homes[HOMES_COLUMNS].sort_values("home_id").reset_index(drop=True)

    halfhourly_path = out_dir / "halfhourly.parquet"
    homes_path = out_dir / "homes.parquet"

    table = pa.Table.from_pandas(halfhourly, preserve_index=False)
    pq.write_table(table, halfhourly_path, row_group_size=row_group_size, compression="snappy")
    homes.to_parquet(homes_path, index=False)

    return halfhourly, homes


if __name__ == "__main__":
    hh_df, homes_df = build_all()
    out_dir = _get_processed_dir()
    print(f"Saved {len(hh_df):,} rows to {out_dir / 'halfhourly.parquet'}")
    print(f"Saved {len(homes_df):,} homes to {out_dir / 'homes.parquet'}")
    print(f"Date range: {hh_df['ts'].min()} -> {hh_df['ts'].max()}")
    print(f"Invalid rows: {int((~hh_df['valid']).sum()):,}")
