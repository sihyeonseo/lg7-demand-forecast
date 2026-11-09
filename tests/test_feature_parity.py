"""Check that train and inference generate exactly the same input features.

Usage: python tests/test_feature_parity.py --train data/train.csv
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import FEATURES, load_training, make_train_samples, make_inference_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    args = p.parse_args()
    sales, menu_info = load_training(args.train)
    cutoff = sales.index[-8]
    train_features = make_train_samples(sales, pd.DatetimeIndex([cutoff]), menu_info)
    inference_features = make_inference_samples(sales.loc[:cutoff].tail(28), menu_info)
    assert np.array_equal(train_features["menu_id"], inference_features["menu_id"])
    assert np.array_equal(train_features["target_date"], inference_features["target_date"])
    for col in FEATURES:
        a = train_features[col].to_numpy(dtype=float)
        b = inference_features[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Mismatch in {col}: max abs diff {np.max(np.abs(a-b))}")
    print(f"PASS: {len(FEATURES)} features match for cutoff {cutoff.date()}")


if __name__ == "__main__":
    main()
