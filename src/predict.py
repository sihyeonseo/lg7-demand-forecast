"""LG Aimers 7th menu demand forecasting.

Requires private competition data to be supplied locally; no TEST sample chaining.
Code reconstructed from the documented successful experiment. It was not run end-to-end
on all ten original TEST files in this environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = [
    "horizon", "weekday", "month", "mean7", "mean28", "std7",
    "zero_ratio28", "positive_days28", "weekday_mean4", "lag7", "lag14",
]


def load_training(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"영업일자", "영업장명_메뉴명", "매출수량"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing training columns: {required - set(df.columns)}")
    df["영업일자"] = pd.to_datetime(df["영업일자"])
    if df.duplicated(["영업일자", "영업장명_메뉴명"]).any():
        raise ValueError("Duplicate date / menu pairs in train.csv")
    menu_info = pd.DataFrame({"full_menu": sorted(df["영업장명_메뉴명"].unique())})
    if not menu_info["full_menu"].str.contains("_", regex=False).all():
        raise ValueError("Menu names must have store prefix separated by underscore")
    menu_info[["store_name", "item_name"]] = menu_info["full_menu"].str.split("_", n=1, expand=True)
    menu_info["menu_id"] = np.arange(len(menu_info), dtype=int)
    menu_info["store_id"] = pd.factorize(menu_info["store_name"], sort=True)[0]
    sales = df.pivot(index="영업일자", columns="영업장명_메뉴명", values="매출수량").sort_index()
    sales = sales[menu_info["full_menu"]]
    if sales.isna().any().any():
        raise ValueError("Missing sales values. Review date/menu panel completeness.")
    if not (sales.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Training dates are not consecutive")
    return sales, menu_info


def feature_blocks(hist: np.ndarray, cutoff: pd.Timestamp, menu_info: pd.DataFrame) -> pd.DataFrame:
    """Generate ordered features: 7 horizon blocks x N menus from exactly 28 days."""
    n_menus = len(menu_info)
    if hist.shape != (28, n_menus) or not np.isfinite(hist).all():
        raise ValueError("Expect finite history of shape (28, number of menus)")
    mean7 = hist[-7:].mean(axis=0)
    mean28 = hist.mean(axis=0)
    std7 = hist[-7:].std(axis=0, ddof=0)
    zero_ratio28 = (hist == 0).mean(axis=0)
    positive_days28 = (hist > 0).sum(axis=0)
    weekday_mean4 = hist.reshape(4, 7, n_menus).mean(axis=0)
    blocks = []
    for h in range(1, 8):
        target_date = cutoff + pd.Timedelta(days=h)
        blocks.append(pd.DataFrame({
            "cutoff": cutoff,
            "target_date": target_date,
            "menu_id": menu_info["menu_id"].to_numpy(),
            "store_id": menu_info["store_id"].to_numpy(),
            "horizon": h,
            "weekday": target_date.dayofweek,
            "month": target_date.month,
            "mean7": mean7,
            "mean28": mean28,
            "std7": std7,
            "zero_ratio28": zero_ratio28,
            "positive_days28": positive_days28,
            "weekday_mean4": weekday_mean4[h - 1],
            "lag7": hist[20 + h],
            "lag14": hist[13 + h],
        }))
    return pd.concat(blocks, ignore_index=True)


def make_train_samples(sales: pd.DataFrame, cutoffs: pd.DatetimeIndex, menu_info: pd.DataFrame) -> pd.DataFrame:
    dates = sales.index
    values = sales.to_numpy(dtype=float)
    blocks = []
    for cutoff in pd.to_datetime(cutoffs):
        idx = dates.get_loc(cutoff)
        if idx < 27 or idx + 7 >= len(dates):
            raise ValueError(f"Not enough observed history/labels for cutoff {cutoff}")
        df = feature_blocks(values[idx - 27:idx + 1], cutoff, menu_info)
        df["target"] = values[idx + 1:idx + 8].reshape(-1)
        blocks.append(df)
    if not blocks:
        raise ValueError("Training cutoffs are empty")
    return pd.concat(blocks, ignore_index=True)


def make_inference_samples(history: pd.DataFrame, menu_info: pd.DataFrame) -> pd.DataFrame:
    if len(history) != 28 or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("History must contain 28 unique, ordered dates")
    if not (history.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("History must span consecutive days")
    names = menu_info["full_menu"].tolist()
    if set(history.columns) != set(names):
        raise ValueError("TEST menus differ from training menus")
    return feature_blocks(history[names].to_numpy(dtype=float), history.index[-1], menu_info)


def fit_model(train: pd.DataFrame, zero_weight: float) -> lgb.LGBMRegressor:
    if not 0 <= zero_weight <= 1:
        raise ValueError("zero_weight must be between 0 and 1")
    weights = np.where(train["target"].to_numpy() == 0, zero_weight, 1.0)
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=300, learning_rate=0.05,
        num_leaves=15, max_depth=5, min_child_samples=50,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(train[FEATURES], train["target"], sample_weight=weights)
    return model


def discover_tests(folder: Path) -> dict[str, Path]:
    tests = {}
    for path in folder.glob("TEST_*.csv"):
        matched = re.fullmatch(r"TEST_(\d{2})(?:\(\d+\))?\.csv", path.name)
        if not matched:
            continue
        key = f"TEST_{matched.group(1)}"
        if key in tests:
            raise ValueError(f"Duplicate TEST id: {key}")
        tests[key] = path
    return dict(sorted(tests.items()))


def load_test(path: Path, menu_info: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"영업일자", "영업장명_메뉴명", "매출수량"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Bad TEST columns: {path}")
    raw["영업일자"] = pd.to_datetime(raw["영업일자"])
    if raw.duplicated(["영업일자", "영업장명_메뉴명"]).any():
        raise ValueError(f"Duplicate TEST records: {path}")
    history = raw.pivot(index="영업일자", columns="영업장명_메뉴명", values="매출수량").sort_index()
    names = menu_info["full_menu"].tolist()
    if set(history.columns) != set(names):
        raise ValueError(f"Menu mismatch: {path}")
    history = history[names]
    if history.isna().any().any():
        raise ValueError(f"Missing TEST values: {path}")
    return history


def build_submission(model, test_files: dict[str, Path], menu_info: pd.DataFrame, sample_submission: Path):
    if set(test_files) != {f"TEST_{i:02d}" for i in range(10)}:
        raise ValueError("Expected exactly TEST_00 through TEST_09")
    names = menu_info["full_menu"].tolist()
    frames = []
    for test_id, path in test_files.items():
        feats = make_inference_samples(load_test(path, menu_info), menu_info)
        pred = np.maximum(model.predict(feats[FEATURES]), 0)
        rows = pd.DataFrame({
            "영업일자": [f"{test_id}+{int(h)}일" for h in feats["horizon"]],
            "영업장명_메뉴명": [names[int(i)] for i in feats["menu_id"]],
            "pred": pred,
        })
        frames.append(rows)
        print(f"Completed: {test_id}")
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot(index="영업일자", columns="영업장명_메뉴명", values="pred")
    sample = pd.read_csv(sample_submission, encoding="utf-8-sig")
    if sample.columns[0] != "영업일자" or set(sample.columns[1:]) != set(names):
        raise ValueError("sample_submission columns not aligned with training")
    if set(sample["영업일자"]) != set(wide.index):
        raise ValueError("sample_submission target rows mismatch")
    result = sample.copy()
    result.iloc[:, 1:] = wide.loc[sample["영업일자"], sample.columns[1:]].to_numpy()
    preds = result.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(preds).all() or (preds < 0).any():
        raise ValueError("Submission predictions must be finite and nonnegative")
    return result, long


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--zero-weight", type=float, default=0.25)
    args = parser.parse_args()

    sales, menu_info = load_training(args.train)
    last = len(sales) - 1
    cutoffs = sales.index[list(range(27, last - 6, 7))]
    train = make_train_samples(sales, cutoffs, menu_info)
    if train["target_date"].max() > sales.index[-1]:
        raise AssertionError("Future label leakage")
    model = fit_model(train, args.zero_weight)
    submission, long = build_submission(model, discover_tests(args.test_dir), menu_info, args.sample_submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, encoding="utf-8-sig")
    long.to_csv(args.output.with_name("test_predictions_long.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved {submission.shape} submission to {args.output}")


if __name__ == "__main__":
    main()
