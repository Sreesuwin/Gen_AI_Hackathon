"""Shared settings. Change values here only, never hard-code them in modules.

Any value can be overridden with an environment variable of the same name,
e.g.  OLLAMA_URL=http://172.22.66.56:11434 streamlit run src/app/streamlit_app.py
"""
import os
from pathlib import Path


def _env(name, default, cast=str):
    val = os.environ.get(name)
    return cast(val) if val is not None else default


ROOT = Path(__file__).resolve().parent.parent

# ---- tariffs and costs (₹, assumed defaults, editable in the UI) ----
IMPORT_TARIFF = _env("IMPORT_TARIFF", 8.0, float)          # ₹ per kWh bought from the grid
EXPORT_TARIFF = _env("EXPORT_TARIFF", 3.0, float)          # ₹ per kWh exported (net-metering credit)
BATTERY_COST_PER_KWH = _env("BATTERY_COST_PER_KWH", 30000, float)   # ₹ installed
PV_COST_PER_KW = _env("PV_COST_PER_KW", 50000, float)               # ₹ installed
BATTERY_EFFICIENCY = _env("BATTERY_EFFICIENCY", 0.90, float)        # round trip
EV_KWH_PER_KM = 0.15
CURRENCY = "₹"

# ---- reference values from our dataset analysis (FY 2012-13, 300 homes) ----
DATASET_MEDIAN_BASELOAD_KW = 0.26
DATASET_P90_BASELOAD_KW = 0.78

# ---- LLM (Ollama runs only on Sreesuwin's RTX 4070 laptop) ----
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:7b")
LLM_FALLBACK_MODEL = "qwen2.5:3b"
LLM_TEMPERATURE = 0.2
MAX_TOOL_ROUNDS = 4

# ---- paths ----
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_FILE = ROOT / "data" / "sample" / "sample_home.parquet"
HALFHOURLY_FILE = PROCESSED_DIR / "halfhourly.parquet"
HOMES_FILE = PROCESSED_DIR / "homes.parquet"
RAW_FILES = {
    "2010-2011": RAW_DIR / "Solar home 2010-2011.csv",
    "2011-2012": RAW_DIR / "Solar home 2011-2012.csv",
    "2012-2013": RAW_DIR / "Solar home 2012-2013.csv",
}
DATE_FORMATS = {"2010-2011": "%d-%b-%y", "2011-2012": "%d/%m/%Y", "2012-2013": "%d/%m/%Y"}
