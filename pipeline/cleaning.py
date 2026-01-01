import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop duplicados
    df = df.drop_duplicates()

    # Padronizar strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.strip()

    # Exemplo: emp_length
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .str.replace("years", "")
            .str.replace("year", "")
            .str.replace("< 1", "0")
            .str.replace("10+", "10")
            .astype(float)
        )

    return df