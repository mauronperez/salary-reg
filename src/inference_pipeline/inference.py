"""
Inference pipeline for Housing Regression MLE.

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing + feature engineering using saved encoders.
- Aligns features with training.
- Returns predictions.
"""

# Raw → preprocess → feature engineering → align schema → model.predict → predictions.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import rename_cols, drop_duplicates,generating_income
from src.feature_pipeline.feature_engineering import mapping_columns, drop_unused_columns, label_encoding

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "income"]  # excluding income column
else:
    TRAIN_FEATURE_COLUMNS = None


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
) -> pd.DataFrame:
    # Step 1: Preprocess raw input

    input_df = rename_cols(input_df)
    input_df = drop_duplicates(input_df)


    # Step 2: Feature engineering

    input_df = mapping_columns(input_df)


    # Step 3: Encodings ----------------


    # Step 4: Separate actuals if present
    y_true = None
    if "income" in input_df.columns:
        y_true = input_df["income"].tolist()
        input_df = input_df.drop(columns=["income"])

    # Step 5: Align columns with training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        input_df = input_df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Step 6: Load model & predict
    model = load(model_path)
    preds = model.predict(input_df)

    # Step 7: Build output
    out = input_df.copy()
    out["predicted_income"] = preds
    if y_true is not None:
        out["actual_income"] = y_true

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
