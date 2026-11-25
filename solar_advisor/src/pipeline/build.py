"""Build data/processed/halfhourly.parquet and homes.parquet from the 3 raw CSVs.   OWNER: Shashikiran

Run once:  python -m src.pipeline.build

Steps (see README section 5.1):
 1. pd.read_csv(path, skiprows=1, keep_default_na=False)   # line 1 is a title; keep "NA" as text
 2. drop rows where Row Quality == "NA" (column only in 2011-12 and 2012-13)
 3. reshape wide -> long -> contract schema   (scripts/make_sample.py:reshape already does this for one home)
 4. concat the 3 years, mark valid = gen_kwh <= pv_kw * 0.5 * 1.05
 5. save halfhourly.parquet (~15 M rows) and homes.parquet (home_id, pv_kw, postcode, has_cl, n_days)
"""


def main():
    raise NotImplementedError("build pipeline: Shashikiran (start from scripts/make_sample.py:reshape)")


if __name__ == "__main__":
    main()
