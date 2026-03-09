"""
Load & time-split the raw dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str = "data/raw/ds_mujer_individual_full.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    df = pd.read_csv(raw_path)

    # Ensure datetime + sort
    df = df[(df['INGRESOS'] > 1)& (df['TRABAJAACT'] == 1)]

    # Splits
    # 80% train, 20% temp
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # Split temp into 10% eval, 10% holdout
    eval_df, holdout_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
