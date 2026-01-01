import pandas as pd
import numpy as np

def parse_emp_length(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if x in ["n/a", "none"]:
        return np.nan
    x = x.replace("years", "").replace("year", "").replace("+", "").strip()
    try:
        return int(x)
    except:
        return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].apply(parse_emp_length)

    return df