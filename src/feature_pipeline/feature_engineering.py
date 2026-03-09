"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump #joblib.dump saves encoders/mappings to disk (important for reusing at inference).

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    drop_cols = ['age','studies','reg_living','ever_married', 'number_children', 'number_living','type_contract','income']
    train = train[drop_cols]
    eval = eval[drop_cols]
    return train, eval

def mapping_columns(df):

    regimen_vivienda_map = {
    2: 1,
    3: 2,
    1: 2,
    4: 3,
    5: 3,
    6: 3
}

    df['reg_living'] = df['reg_living'].map(regimen_vivienda_map)

    ec_binary_map = {
    2: 1,
    3: 1,
    5: 1,
    1: 0,
    4: 0
}

    df['ever_married'] = df['ever_married'].map(ec_binary_map)

    return df 

def label_encoding(df: pd.DataFrame):
    cols = ['studies', 'reg_living', 'type_contract']
    le = LabelEncoder()
    
    for col in cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
        else:
            print(f"This {col} is not included in the dataframe. ")
    
    return df





# ---------- pipeline ----------

#Handles full pipeline: 
#reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)


    # Date features
    train_df = mapping_columns(train_df)
    eval_df = mapping_columns(eval_df)
    holdout_df = mapping_columns(holdout_df)


    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    train_df = label_encoding(train_df)
    eval_df = label_encoding(eval_df)
    holdout_df = label_encoding(holdout_df)




    # Save engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    run_feature_engineering()
