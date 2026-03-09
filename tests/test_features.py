import pandas as pd
import pytest
from pathlib import Path



from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.preprocess import (drop_duplicates, preprocess_split
)
from src.feature_pipeline.feature_engineering import (drop_unused_columns, run_feature_engineering
)

# =========================
# load.py – unit test
# =========================
# Confirms time-based splitting works.


def test_drop_duplicates_removes_dupes():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "year": [2020, 2020],
        "median_list_price": [100, 100]
    })
    cleaned = drop_duplicates(df)
    assert cleaned.shape[0] == 1
    print("✅ Duplicate removal test passed")

