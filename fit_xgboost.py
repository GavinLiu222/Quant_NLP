"""Fit XGBoost classifiers that predict, from per-article text representations,
whether a stock rises between entry and the same trading day's close.

Input
  outputs/alpaca_news_finbert_representations.parquet
    symbols, created_at      ticker and publish time (UTC), the join key
    logits                   3-d raw scores of a positive / negative / neutral
                             classification head (order in the schema metadata
                             under `logit_labels`)
    cls_last, pooler, mean_last, cls_second_last, mean_second_last
                             768-d sentence vectors
    Only `symbols`, `created_at` and the columns named in REPRESENTATIONS are read.
  /Volumes/GAVIN/QuantDB/high_freq.duckdb
    One table of 1-minute bars per trading day, named yyyy-mm-dd; the table
    names are the trading calendar.  Bars run 04:00-20:00 New York time, and
    only the regular session is used: [09:30, 16:00), or [09:30, 13:00) on the
    half days in EARLY_CLOSE_DAYS.
    ticker, window_start     ticker and bar start (UTC nanoseconds)
    open, close              first and last trade price in the bar

Labels
  Each article goes to trading day T, the first trading day whose regular
  session closes strictly after the publish time.  T-1 is the previous trading
  day, so weekend and holiday articles go to the next trading day.
    pre_open   published between T-1's close and T's open: enter at T's open
    intraday   published during T's session: enter at the first bar starting
               at or after the publish time
  Both cases are one rule: entry_price is the open of T's first regular-session
  bar with window_start >= publish time, exit_price is the close of T's last
  regular-session bar.  Articles with no such bar on T are dropped.
  target_return = exit_price / entry_price - 1, left blank when entry_price <= 0.
  label = 1 (up) if target_return > 0, else 0.  Rows with no return are dropped,
  and articles sharing one trade (same ticker, trade_date, entry_time) are
  collapsed to a single row, chosen by KEEP_PER_ENTRY.

Split
  Chronological and aligned to date boundaries: the first 70% of rows train, the
  last 30% test.  The tail 20% of the training rows is the validation set for
  early stopping and parameter selection; the test set is never used for either.

Model
  One XGBoost per representation.  Every combination in PARAM_GRID is fitted with
  early stopping on validation logloss; the combination with the lowest logloss
  is refitted on the whole training set with its early-stopped number of trees,
  then scored on the test set.

Output
  console: per-representation test results -- chosen parameters, number and
    share of predicted ups, up precision, accuracy -- against the test set's
    base rate of ups
  outputs/xgb_grid_results.csv
    representation, learning_rate, max_depth, subsample, best_iteration,
    valid_logloss -- one row per (representation, parameter combination)
"""
import itertools
import re
import time
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

PARQUET_PATH = '/Users/gavin/Programming/US_Stock_QuantTrading/outputs/alpaca_news_finbert_representations.parquet'
PRICES_PATH = '/Volumes/GAVIN/QuantDB/high_freq.duckdb'
MARKET_TIMEZONE = 'America/New_York'
SESSION_OPEN = '09:30'
SESSION_CLOSE = '16:00'                 # the 16:00 bar is already after-hours trading, not the close
EARLY_SESSION_CLOSE = '13:00'
EARLY_CLOSE_DAYS = {                    # NYSE 13:00 half days; add new ones when the DB gets new data
    '2021-11-26', '2022-11-25', '2023-07-03', '2023-11-24', '2024-07-03', '2024-11-29',
    '2024-12-24', '2025-07-03', '2025-11-28', '2025-12-24', '2026-11-27', '2026-12-24',
}
KEEP_PER_ENTRY = 'last'
LABEL_LOWER_Q = 0.3
LABEL_UPPER_Q = 0.7
REPRESENTATIONS = ['logits']    # REPRESENTATIONS = ['cls_last', 'pooler', 'mean_last', 'cls_second_last', 'mean_second_last']
TEST_SIZE = 0.3                 # train : test = 7 : 3
VALID_SIZE = 0.2
PARAM_GRID = {
    'learning_rate': [0.025, 0.05, 0.075, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.85, 0.9, 0.95],
}
FIXED_PARAMS = {
    'objective': 'binary:logistic',     # binary; for 3 classes use multi:softprob + num_class=3
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
}
EVAL_METRIC = 'logloss'                 # validation metric for tuning and early stopping; follows the objective
N_ESTIMATORS = 500                      # upper bound; the actual number of rounds is set by early stopping
EARLY_STOPPING_ROUNDS = 30
GRID_RESULTS_PATH = '/Users/gavin/Programming/US_Stock_QuantTrading/outputs/xgb_grid_results.csv'

# --------------------------------------------------------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------------------------------------------------------

df_features = pd.read_parquet(PARQUET_PATH, columns=['symbols', 'created_at'] + REPRESENTATIONS)

n_raw = len(df_features)
df_features = df_features.dropna(subset=['created_at', 'symbols']).rename(columns={'symbols': 'ticker'})
print(f'feature table: {n_raw:,} rows -> {len(df_features):,} after dropping nulls')


def to_utc_ns(days, clocks):
    """New York wall-clock times on the given days as UTC nanoseconds, the unit window_start is stored in."""
    local = pd.to_datetime([f'{day} {clock}' for day, clock in zip(days, clocks)]).tz_localize(MARKET_TIMEZONE)
    return local.tz_convert('UTC').as_unit('ns').asi8


# Entry and exit depend only on (ticker, publish time); articles on one stock at the same instant share them
news = df_features[['ticker', 'created_at']].drop_duplicates().reset_index(drop=True)
news['created_ns'] = news['created_at'].dt.as_unit('ns').astype('int64')

con = duckdb.connect(PRICES_PATH, read_only=True)

# Trading calendar: the DB has one table per trading day and none for weekends or holidays
days = sorted(name for (name,) in con.execute('SELECT table_name FROM duckdb_tables()').fetchall()
              if re.fullmatch(r'\d{4}-\d{2}-\d{2}', name))
open_ns = to_utc_ns(days, [SESSION_OPEN] * len(days))
close_ns = to_utc_ns(days, [EARLY_SESSION_CLOSE if day in EARLY_CLOSE_DAYS else SESSION_CLOSE for day in days])

# T = the first trading day whose close is strictly after the publish time, so everything from T-1's close
# (16:00:00 itself included) through T's session belongs to T
pos = np.searchsorted(close_ns, news['created_ns'].to_numpy(), side='right')
# Articles after the DB's last close have no T; articles before its first open have no known T-1 close
in_range = (pos < len(days)) & ((pos > 0) | (news['created_ns'].to_numpy() >= open_ns[0]))
print(f'minute bars cover {days[0]} ~ {days[-1]}: {(~in_range).sum():,} of {len(news):,} '
      f'(ticker, created_at) pairs fall outside and are dropped')
news = news[in_range].reset_index(drop=True)
pos = pos[in_range]
news['trade_date'] = np.asarray(days)[pos]
news['session'] = np.where(news['created_ns'].to_numpy() < open_ns[pos], 'pre_open', 'intraday')

# One query per trading day.  The ASOF JOIN finds the first regular-session bar starting at or after the
# publish time, which for pre-open articles is T's first bar.  Both joins are inner: a pair with no bar
# after its publish time on T (halted, or published in the session's last minute) is dropped
con.register('news', news[['ticker', 'created_ns', 'trade_date']])
parts = []
for i in tqdm(np.unique(pos), desc='reading minute bars', unit='day'):
    day = days[i]
    parts.append(con.execute(f"""
        WITH bars AS (
            SELECT ticker, window_start, open, close
            FROM "{day}"
            WHERE window_start >= {open_ns[i]} AND window_start < {close_ns[i]}
              AND ticker IN (SELECT ticker FROM news WHERE trade_date = '{day}')
        ),
        last_bar AS (
            SELECT ticker, max(window_start) AS exit_ns, arg_max(close, window_start) AS exit_price
            FROM bars GROUP BY ticker
        )
        SELECT n.ticker, n.created_ns, b.window_start AS entry_ns, b.open AS entry_price, l.exit_ns, l.exit_price
        FROM (SELECT * FROM news WHERE trade_date = '{day}') n
        ASOF JOIN bars b ON n.ticker = b.ticker AND n.created_ns <= b.window_start
        JOIN last_bar l ON n.ticker = l.ticker
    """).fetchdf())
con.close()

df_target = news.merge(pd.concat(parts, ignore_index=True), on=['ticker', 'created_ns'], validate='one_to_one')
print(f'{len(df_target):,} of {len(news):,} pairs have a bar on T at or after publish time; the rest are dropped')
df_target['trade_date'] = pd.to_datetime(df_target['trade_date'])
for col in ['entry', 'exit']:
    df_target[f'{col}_time'] = pd.to_datetime(df_target.pop(f'{col}_ns'), unit='ns', utc=True)
df_target = df_target[['ticker', 'created_at', 'trade_date', 'session',
                       'entry_time', 'entry_price', 'exit_time', 'exit_price']]
df_target['target_return'] = df_target['exit_price'] / df_target['entry_price'] - 1
df_target['target_return'] = df_target['target_return'].where(df_target['entry_price'] > 0)  # an entry price of 0 is bad data; no return
print(df_target.head(10))

# Left join: every df_features row is kept.  One stock can have several articles at the same instant,
# but only one row in the price table
df = df_features.merge(df_target, on=['ticker', 'created_at'], how='left', validate='many_to_one')
n_unmatched = df['target_return'].isna().sum()
print(f'merged: {len(df_features)} feature rows -> {len(df)} rows, '
      f'{n_unmatched} of them with no matching (ticker, created_at) in df_target')

# Rows without a label cannot be trained on; drop them
df = df.dropna(subset=['target_return'])
print(f'after dropping unlabeled rows: {len(df)} rows')

# Articles with the same entry point have identical labels: everything published before T's open enters
# at the open, and intraday articles published within the same minute land on the same bar.
# Keep one article per trade; stable-sort by the raw timestamp first so KEEP_PER_ENTRY's first/last is well defined
df = df.sort_values('created_at', kind='mergesort')
n_before = len(df)
df = df.drop_duplicates(subset=['ticker', 'trade_date', 'entry_time'], keep=KEEP_PER_ENTRY)
df = df.rename(columns={'trade_date': 'date'}).reset_index(drop=True)
print(f'one article per trade: {n_before} -> {len(df)} rows')
print(df['session'].value_counts().rename({'pre_open': 'pre-open (enter at T open)', 'intraday': 'intraday (enter after publish)'}))
print(df.head(10))

# Return label: cut target_return into three buckets by its full-sample quantiles
# q_low, q_high = df['target_return'].quantile([LABEL_LOWER_Q, LABEL_UPPER_Q])
# assert q_low < q_high, f'quantile cut-offs coincide ({q_low} / {q_high}); target_return is too concentrated'

# # top 30% (>= q_high) up 1; bottom 30% (<= q_low) down -1; middle 40% flat 0
# df['label'] = np.select(
#     [df['target_return'] >= q_high, df['target_return'] <= q_low],
#     [1, -1],
#     default=0,
# ).astype('int8')

# print(f'label cut-offs: down <= {q_low:.4%} < flat < {q_high:.4%} <= up')
# print(df['label'].value_counts().reindex([1, 0, -1]).rename({1: 'up(1)', 0: 'flat(0)', -1: 'down(-1)'}))

# Binary label: 1 (up) if the return from entry to T's close is positive, otherwise 0 (not up,
# including rows whose return is exactly 0)
df['label'] = (df['target_return'] > 0).astype('int8')

print(df['label'].value_counts().reindex([1, 0]).rename({1: 'up(1)', 0: 'not up(0)'}))
print(f"share of ups: {df['label'].mean():.2%}")

# --------------------------------------------------------------------------------------------------------------
# Model Fitting
# --------------------------------------------------------------------------------------------------------------

# The label already is the 0/1 encoding XGBoost expects; no remapping needed
UP_CLASS = 1

# Feature matrices come straight from df, so df_features / df_target are no longer needed; free them now
# (the 768-d float32 sentence vectors take a lot of memory)
del df_features, df_target

# Split chronologically, not at random: samples from the same day or adjacent trading days are highly
# correlated, and a random split would put "the neighbouring article from the same day" into both train
# and test, letting the model infer that day's market move and inflating the measured precision.
df = df.sort_values('date', kind='mergesort').reset_index(drop=True)
y = df['label'].to_numpy()


def split_before(dates, tail_frac):
    """Split rows into head and tail by tail_frac and return the head's boolean mask. The cut is aligned to a date boundary, so no day is split."""
    cut = int(len(dates) * (1 - tail_frac))
    return (dates < dates.iloc[cut]).to_numpy()


is_train = split_before(df['date'], TEST_SIZE)                 # first 70% -> train, last 30% -> test
is_test = ~is_train
# Split the training set once more: the earlier part is fitted on, the later part validates
# (early stopping + parameter selection); the test set takes no part
train_pos = np.flatnonzero(is_train)
is_fit = np.zeros(len(df), dtype=bool)
is_fit[train_pos[split_before(df.loc[is_train, 'date'], VALID_SIZE)]] = True
is_valid = is_train & ~is_fit

for name, mask in [('fit', is_fit), ('valid', is_valid), ('test', is_test)]:
    d = df.loc[mask, 'date']
    print(f'{name:<5} set: {mask.sum():>7,} rows ({mask.mean():.1%})  {d.min():%Y-%m-%d} ~ {d.max():%Y-%m-%d}')

y_fit, y_valid, y_train, y_test = y[is_fit], y[is_valid], y[is_train], y[is_test]
grid = list(itertools.product(*PARAM_GRID.values()))
print(f'\ngrid search over {len(grid)} parameter combinations per representation, {len(REPRESENTATIONS)} representation(s)\n')

grid_rows = []
summary_rows = []
for rep in REPRESENTATIONS:
    t0 = time.time()
    X = np.vstack(df[rep].to_numpy())        # (n_samples, 768); (n_samples, 3) for logits
    X_fit, X_valid, X_train, X_test = X[is_fit], X[is_valid], X[is_train], X[is_test]

    # Select parameters on validation logloss (exactly what the objective optimizes), not on up precision:
    # that metric can be pushed high by predicting very few ups, which would pick a model that almost never trades.
    best = None
    for lr, depth, sub in grid:
        model = xgb.XGBClassifier(
            **FIXED_PARAMS, learning_rate=lr, max_depth=depth, subsample=sub,
            n_estimators=N_ESTIMATORS, eval_metric=EVAL_METRIC,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
        grid_rows.append({'representation': rep, 'learning_rate': lr, 'max_depth': depth,
                          'subsample': sub, 'best_iteration': model.best_iteration + 1,
                          'valid_logloss': model.best_score})
        if best is None or model.best_score < best['valid_logloss']:
            best = grid_rows[-1]

    # With the parameters chosen, refit on the whole training set (fit + valid), the number of trees fixed
    # at the early-stopped value
    model = xgb.XGBClassifier(
        **FIXED_PARAMS, learning_rate=best['learning_rate'], max_depth=best['max_depth'],
        subsample=best['subsample'], n_estimators=best['best_iteration'],
    )
    model.fit(X_train, y_train, verbose=False)

    # Performance: of the samples predicted "up", the share that actually went up
    pred = model.predict(X_test)
    picked = pred == UP_CLASS
    precision_up = float((y_test[picked] == UP_CLASS).mean()) if picked.any() else float('nan')
    summary_rows.append({
        'representation': rep,
        'learning_rate': best['learning_rate'],
        'max_depth': best['max_depth'],
        'subsample': best['subsample'],
        'n_trees': best['best_iteration'],
        'valid_logloss': round(best['valid_logloss'], 5),
        'n_pred_up': int(picked.sum()),                      # number of predicted ups; precision means little if this is tiny
        'pred_up_rate': round(float(picked.mean()), 4),
        'precision_up': round(precision_up, 4),
        'accuracy': round(float((pred == y_test).mean()), 4),
    })
    print(f'[{rep}] best params lr={best["learning_rate"]}, max_depth={best["max_depth"]}, '
          f'subsample={best["subsample"]}, n_trees={best["best_iteration"]} | '
          f'up precision {precision_up:.2%} ({picked.sum():,} predicted up) | took {time.time() - t0:.0f}s')
    del X, X_fit, X_valid, X_train, X_test

pd.DataFrame(grid_rows).to_csv(GRID_RESULTS_PATH, index=False)

# Baseline: the share of ups already in the test set.  Only a precision above it means the model picked something out
baseline = float((y_test == UP_CLASS).mean())
summary = pd.DataFrame(summary_rows).sort_values('precision_up', ascending=False)
print(f'\n{"=" * 100}\nperformance per representation (test set, up baseline = {baseline:.2%})\n{"=" * 100}')
print(summary.to_string(index=False))
print(f'\nfull grid results written to {GRID_RESULTS_PATH}')
