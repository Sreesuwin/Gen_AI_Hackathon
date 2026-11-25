"""Create data/sample/sample_home.parquet: one home, one year, in the contract schema.

Run:  python scripts/make_sample.py [home_id] [year]      (defaults: 1 2012-2013)
Shashikiran's full pipeline (src/pipeline/build.py) does the same for all homes and years.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def reshape(raw: pd.DataFrame, date_format: str) -> pd.DataFrame:
    """Wide Ausgrid rows (one per home/category/day, 48 slots) -> contract schema."""
    slots = list(raw.columns[5:53])                      # '0:30' ... '23:30', '0:00'
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], format=date_format)
    long = raw.melt(id_vars=["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"],
                    value_vars=slots, var_name="slot", value_name="kwh")
    idx = long["slot"].map({s: i for i, s in enumerate(slots)})
    long["ts"] = long["date"] + pd.to_timedelta(idx * 30, unit="min")   # slot i starts at i*30 min
    wide = long.pivot_table(index=["Customer", "Generator Capacity", "ts"], columns="Consumption Category",
                            values="kwh", aggfunc="first").reset_index()
    for c in ("GG", "GC", "CL"):
        if c not in wide:
            wide[c] = 0.0
    out = pd.DataFrame({
        "home_id": wide["Customer"].astype(int),
        "ts": wide["ts"],
        "gen_kwh": wide["GG"].fillna(0.0),
        "gc_kwh": wide["GC"].fillna(0.0),
        "cl_kwh": wide["CL"].fillna(0.0),
    })
    pv_kw = wide["Generator Capacity"].astype(float)
    out["cons_kwh"] = out["gc_kwh"] + out["cl_kwh"]
    out["self_kwh"] = out[["gen_kwh", "cons_kwh"]].min(axis=1)
    out["import_kwh"] = (out["cons_kwh"] - out["gen_kwh"]).clip(lower=0)
    out["export_kwh"] = (out["gen_kwh"] - out["cons_kwh"]).clip(lower=0)
    out["valid"] = out["gen_kwh"] <= pv_kw * 0.5 * 1.05
    return out.sort_values(["home_id", "ts"]).reset_index(drop=True)


def main(home_id: int = 1, year: str = "2012-2013"):
    # keep_default_na=False: otherwise pandas turns the literal "NA" quality flag into NaN,
    # which looks the same as an empty (good) row
    raw = pd.read_csv(config.RAW_FILES[year], skiprows=1, keep_default_na=False)
    if "Row Quality" in raw.columns:
        raw = raw[raw["Row Quality"] != "NA"]
    raw = raw[raw["Customer"] == home_id]
    df = reshape(raw, config.DATE_FORMATS[year])
    config.SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SAMPLE_FILE, index=False)
    meta = raw.iloc[0]
    print(f"home {home_id} ({meta['Generator Capacity']} kW, postcode {meta['Postcode']}), {year}: "
          f"{len(df)} rows, {(~df.valid).sum()} invalid -> {config.SAMPLE_FILE}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, sys.argv[2] if len(sys.argv) > 2 else "2012-2013")
