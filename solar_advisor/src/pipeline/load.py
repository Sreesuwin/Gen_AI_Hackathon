"""Loading data in the contract schema.   OWNER: Shashikiran

Until data/processed/halfhourly.parquet exists, every loader falls back to the sample
file (home 1, FY 2012-13) so the rest of the team can build against real-shaped data.
"""
import pandas as pd

from src import config
from src.contracts import HALFHOURLY_COLUMNS


def _processed_ready() -> bool:
    return config.HALFHOURLY_FILE.exists() and config.HOMES_FILE.exists()


def list_homes() -> pd.DataFrame:
    """Columns: home_id, pv_kw, postcode, has_cl, n_days."""
    if _processed_ready():
        return pd.read_parquet(config.HOMES_FILE)
    # STUB until the pipeline runs: only the sample home
    df = pd.read_parquet(config.SAMPLE_FILE)
    return pd.DataFrame([{"home_id": 1, "pv_kw": 3.78, "postcode": 2076,
                          "has_cl": bool(df.cl_kwh.sum() > 0), "n_days": df.ts.dt.date.nunique()}])


def load_home(home_id: int, start=None, end=None) -> pd.DataFrame:
    """Half-hourly rows for one home, valid rows only, sorted by ts."""
    if _processed_ready():
        df = pd.read_parquet(config.HALFHOURLY_FILE, filters=[("home_id", "==", int(home_id))])
    else:
        df = pd.read_parquet(config.SAMPLE_FILE)          # STUB: always home 1
    df = df[df["valid"]]
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] < pd.Timestamp(end)]
    return df[HALFHOURLY_COLUMNS].sort_values("ts").reset_index(drop=True)


def load_user_csv(file, pv_kw: float) -> pd.DataFrame:
    """User upload with columns timestamp, generation_kwh, consumption_kwh -> contract schema, home_id 0.

    TODO (Shashikiran): resample any 15-60 min data to 30 min, validate columns, flag
    readings above pv_kw * 0.5 * 1.05.
    """
    raise NotImplementedError("load_user_csv: Shashikiran")
