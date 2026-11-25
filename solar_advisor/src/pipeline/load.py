"""Load, reshape, and serve Ausgrid half-hourly solar home data.

Follows the M1 Data Pipeline contract in Team_Build_Plan.md (Sections 1.1, 1.3, 2.1, 2.2)
while reading directly from the existing `Dataset` directory without duplicating raw CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "Dataset"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "sample_home.parquet"
DEFAULT_SAMPLE_CSV_NAME = "Solar home 2012-2013.csv"
SAMPLE_HOME_META = {"Customer": [1], "Generator Capacity": [3.78], "Postcode": [2076]}

HALFHOURLY_COLUMNS = [
    "home_id",
    "ts",
    "gen_kwh",
    "gc_kwh",
    "cl_kwh",
    "cons_kwh",
    "self_kwh",
    "import_kwh",
    "export_kwh",
    "valid",
]


def _get_processed_dir() -> Path:
    """Return processed directory, respecting src.config.PROCESSED if present."""
    try:
        from src import config  # type: ignore

        processed = Path(getattr(config, "PROCESSED_DIR", getattr(config, "PROCESSED", DEFAULT_PROCESSED_DIR)))
        return processed if processed.is_absolute() else PROJECT_ROOT / processed
    except ImportError:
        return DEFAULT_PROCESSED_DIR


def resolve_dataset_dir() -> Path:
    """Locate the existing Ausgrid dataset directory without duplicating raw files."""
    candidates = [
        DEFAULT_DATASET_DIR,
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "data" / "raw",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("Solar home *.csv")):
            return candidate
    raise FileNotFoundError(
        "Could not locate Ausgrid CSV files in Dataset/, dataset/, or data/raw/."
    )


def _infer_date_format(date_series: pd.Series) -> str:
    """Infer the Ausgrid CSV date format ('%d/%m/%Y' vs '%d-%b-%y')."""
    sample = str(date_series.dropna().iloc[0]).strip()
    return "%d/%m/%Y" if "/" in sample else "%d-%b-%y"


def _add_energy_balance(df: pd.DataFrame, pv_kw: pd.Series | float) -> pd.DataFrame:
    """Compute cons_kwh, self_kwh, import_kwh, export_kwh, and validity flag."""
    df["cons_kwh"] = df["gc_kwh"] + df["cl_kwh"]
    df["self_kwh"] = np.minimum(df["gen_kwh"], df["cons_kwh"])
    df["import_kwh"] = np.maximum(df["cons_kwh"] - df["gen_kwh"], 0.0)
    df["export_kwh"] = np.maximum(df["gen_kwh"] - df["cons_kwh"], 0.0)
    max_halfhour_gen = pv_kw * 0.5 * 1.05
    df["valid"] = (
        (df["gen_kwh"] <= max_halfhour_gen)
        & (df["gen_kwh"] >= 0.0)
        & (df["gc_kwh"] >= 0.0)
        & (df["cl_kwh"] >= 0.0)
    )
    return df


def reshape_ausgrid_csv(
    csv_path: Union[str, Path],
    home_id: int | None = None,
) -> pd.DataFrame:
    """Read an Ausgrid CSV and reshape its 48 half-hourly slots to schema 1.1.

    Parameters
    ----------
    csv_path:
        Path to one of the Ausgrid ``Solar home YYYY-YYYY.csv`` files.
    home_id:
        Optional customer ID filter applied before reshaping for speed.

    Returns
    -------
    pd.DataFrame
        DataFrame conforming to Section 1.1 of Team_Build_Plan.md.
    """
    path = Path(csv_path)
    raw = pd.read_csv(
        path,
        skiprows=1,
        keep_default_na=False,
        na_values=[""],
        dtype={"Row Quality": str},
    )

    if home_id is not None:
        raw = raw[raw["Customer"] == int(home_id)].copy()

    if "Row Quality" in raw.columns:
        raw = raw[raw["Row Quality"].fillna("") != "NA"].copy()

    if raw.empty:
        return pd.DataFrame(columns=HALFHOURLY_COLUMNS)

    raw["date"] = pd.to_datetime(raw["date"], format=_infer_date_format(raw["date"]))

    slots = list(raw.columns[5:53])
    slot_offsets = {slot: idx * 30 for idx, slot in enumerate(slots)}

    long_df = raw.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"],
        value_vars=slots,
        var_name="slot",
        value_name="kwh",
    )
    long_df["ts"] = long_df["date"] + pd.to_timedelta(
        long_df["slot"].map(slot_offsets), unit="min"
    )
    long_df["kwh"] = pd.to_numeric(long_df["kwh"], errors="coerce")

    pivoted = (
        long_df.pivot_table(
            index=["Customer", "Generator Capacity", "ts"],
            columns="Consumption Category",
            values="kwh",
            aggfunc="sum",
        )
        .reset_index()
    )
    pivoted.columns.name = None

    for cat in ("GG", "GC", "CL"):
        if cat not in pivoted.columns:
            pivoted[cat] = 0.0

    pivoted = pivoted.rename(
        columns={
            "Customer": "home_id",
            "Generator Capacity": "pv_kw",
            "GG": "gen_kwh",
            "GC": "gc_kwh",
            "CL": "cl_kwh",
        }
    )

    for col in ("gen_kwh", "gc_kwh", "cl_kwh"):
        pivoted[col] = pivoted[col].fillna(0.0).astype("float64")

    pivoted = _add_energy_balance(pivoted, pivoted["pv_kw"])
    pivoted["home_id"] = pivoted["home_id"].astype("int64")
    pivoted["valid"] = pivoted["valid"].astype("bool")

    result = (
        pivoted[HALFHOURLY_COLUMNS]
        .sort_values(["home_id", "ts"])
        .reset_index(drop=True)
    )
    return result


def build_sample_home(
    csv_path: Union[str, Path, None] = None,
    output_path: Union[str, Path, None] = None,
    home_id: int = 1,
) -> pd.DataFrame:
    """Extract one home from an Ausgrid CSV and save ``sample_home.parquet``."""
    src_csv = (
        Path(csv_path)
        if csv_path is not None
        else resolve_dataset_dir() / DEFAULT_SAMPLE_CSV_NAME
    )
    out_file = Path(output_path) if output_path is not None else DEFAULT_SAMPLE_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = reshape_ausgrid_csv(src_csv, home_id=home_id)
    df.to_parquet(out_file, index=False)
    return df


def list_homes() -> pd.DataFrame:
    """Return metadata for available homes (``home_id, pv_kw, postcode, has_cl, n_days``)."""
    homes_path = _get_processed_dir() / "homes.parquet"
    if homes_path.is_file():
        return pd.read_parquet(homes_path)

    if DEFAULT_SAMPLE_PATH.is_file():
        sample_df = pd.read_parquet(DEFAULT_SAMPLE_PATH)
        try:
            src_csv = resolve_dataset_dir() / DEFAULT_SAMPLE_CSV_NAME
            meta = pd.read_csv(
                src_csv,
                skiprows=1,
                usecols=["Customer", "Generator Capacity", "Postcode"],
            )
        except FileNotFoundError:
            # Raw CSVs not downloaded: use the known details of the sample home.
            meta = pd.DataFrame(SAMPLE_HOME_META)
        home_ids = sample_df["home_id"].unique()
        meta = meta[meta["Customer"].isin(home_ids)].drop_duplicates("Customer")
        rows = []
        for _, row in meta.iterrows():
            hid = int(row["Customer"])
            sub = sample_df[sample_df["home_id"] == hid]
            rows.append(
                {
                    "home_id": hid,
                    "pv_kw": float(row["Generator Capacity"]),
                    "postcode": int(row["Postcode"]),
                    "has_cl": bool((sub["cl_kwh"] > 0).any()),
                    "n_days": int(sub["ts"].dt.normalize().nunique()),
                }
            )
        return pd.DataFrame(rows)

    raise FileNotFoundError(
        f"Neither {homes_path} nor {DEFAULT_SAMPLE_PATH} exists. "
        "Run src/pipeline/load.py or src/pipeline/build.py first."
    )


def load_home(
    home_id: int,
    start: Union[str, pd.Timestamp, None] = None,
    end: Union[str, pd.Timestamp, None] = None,
) -> pd.DataFrame:
    """Load valid half-hourly readings for ``home_id`` conforming to schema 1.1."""
    hid = int(home_id)
    halfhourly_path = _get_processed_dir() / "halfhourly.parquet"

    if halfhourly_path.is_file():
        df = pd.read_parquet(halfhourly_path, filters=[("home_id", "==", hid)])
    elif DEFAULT_SAMPLE_PATH.is_file() and hid == 1:
        df = pd.read_parquet(DEFAULT_SAMPLE_PATH, filters=[("home_id", "==", hid)])
    else:
        raise FileNotFoundError(
            f"Processed parquet not found at {halfhourly_path} "
            f"(and sample_home.parquet only contains home_id=1)."
        )

    df = df[df["valid"]].copy()

    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if isinstance(end, str) and len(end.strip()) == 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["ts"] <= end_ts]

    return df[HALFHOURLY_COLUMNS].sort_values("ts").reset_index(drop=True)


def load_user_csv(file: Union[str, Path, IO[str], IO[bytes]], pv_kw: float) -> pd.DataFrame:
    """Load a user-uploaded CSV (``timestamp, generation_kwh, consumption_kwh``) to schema 1.1."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"timestamp", "generation_kwh", "consumption_kwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in user CSV: {sorted(missing)}")

    df["ts"] = pd.to_datetime(df["timestamp"])
    df["gen_kwh"] = pd.to_numeric(df["generation_kwh"], errors="coerce").fillna(0.0)
    df["cons_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce").fillna(0.0)

    resampled = (
        df.set_index("ts")[["gen_kwh", "cons_kwh"]]
        .resample("30min", label="left", closed="left")
        .sum()
        .reset_index()
    )

    resampled["home_id"] = np.int64(0)
    resampled["gc_kwh"] = resampled["cons_kwh"].astype("float64")
    resampled["cl_kwh"] = np.float64(0.0)
    resampled = _add_energy_balance(resampled, float(pv_kw))
    resampled["valid"] = resampled["valid"].astype("bool")

    return resampled[HALFHOURLY_COLUMNS].sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    sample_df = build_sample_home(home_id=1)
    print(f"Saved {len(sample_df)} rows to {DEFAULT_SAMPLE_PATH}")
    print(f"Date range: {sample_df['ts'].min()} -> {sample_df['ts'].max()}")
    print(f"Valid rows: {int(sample_df['valid'].sum())} / {len(sample_df)}")
