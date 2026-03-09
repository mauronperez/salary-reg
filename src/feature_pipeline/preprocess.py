"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def rename_cols(df:pd.DataFrame):
    rename_map = {
    'EDAD': 'age',
    'EC': 'ever_married',
    'REGVI': 'reg_living',
    'ESTUDIOSA': 'studies',
    'NHOGAR': 'number_living',
    'NHIJOBIO': 'number_children',
    'JORNADA': 'type_contract',
    'INGRESOS': 'income'}
    
    df.rename(columns=rename_map,inplace=True)

    return df 


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = rename_cols(df)
    df = drop_duplicates(df)
    df = generating_income(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df



def generating_income(df):

    range_income = {
    1: (0, 0),           # No tiene ingresos
    2: (350, 500),         # Menos de 500
    3: (500, 1000),      # De 500 a menos de 1000
    4: (1000, 1500),     # De 1000 a menos de 1500
    5: (1500, 2000),     # De 1500 a menos de 2000
    6: (2000, 2500),     # De 2000 a menos de 2500
    7: (2500, 3000),     # De 2500 a menos de 3000
    8: (3000, 5000),     # De 3000 a menos de 5000
    9: (5000, 7000)      # De 5000 o más (poner límite superior)
    }
    def generate_income(codigo):
        min_val, max_val = range_income[codigo]
        return round(np.random.uniform(min_val, max_val), 2)
    
    df['income'] = df['income'].apply(generate_income)

    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    run_preprocess()
