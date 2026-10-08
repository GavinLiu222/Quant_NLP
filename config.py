"""Paths and device for the whole pipeline -- the only file you need to edit.

Every script imports this module, so pointing a path somewhere else here moves
it for all four steps.  The defaults keep everything inside the repository:

    data/alpaca_news_tagged.csv     tagged Alpaca news            (you provide)
    data/high_freq.duckdb           1-minute bars, a table a day  (you provide)
    models/finbert/                 FinBERT checkpoint            (you provide)
    outputs/                        everything the pipeline writes

Any path can be absolute, e.g. PRICES_DB = Path('/mnt/quant_db/high_freq.duckdb');
keep it a Path, since the scripts call .parent / .mkdir on the output paths.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Inputs -- put your files here, or point these at where they already live
# --------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / 'data'
NEWS_TAGGED_CSV = DATA_DIR / 'alpaca_news_tagged.csv'      # read by preprocessing.py
PRICES_DB = DATA_DIR / 'high_freq.duckdb'                   # read by fit_xgboost.py
MODEL_DIR = PROJECT_ROOT / 'models' / 'finbert'             # tokenizer + weights, read by tokenization.py and bert.py

# --------------------------------------------------------------------------
# Outputs -- each step's output is the next step's input
# --------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
NEWS_FILTERED_CSV = OUTPUT_DIR / 'alpaca_news_tagged_filtered.csv'               # preprocessing.py -> tokenization.py
NEWS_TOKENIZED_CSV = OUTPUT_DIR / 'alpaca_news_tagged_filtered_tokenized.csv'    # tokenization.py -> bert.py
FINBERT_PARQUET = OUTPUT_DIR / 'alpaca_news_finbert_representations.parquet'     # bert.py -> fit_xgboost.py
XGB_GRID_RESULTS_CSV = OUTPUT_DIR / 'xgb_grid_results.csv'                       # fit_xgboost.py

# --------------------------------------------------------------------------
# Device for FinBERT inference: 'auto' picks cuda, then mps, then cpu
# --------------------------------------------------------------------------
DEVICE = 'auto'
