import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

from pipeline.cleaning import clean_data
from pipeline.features import build_features


# =========================
# CONFIG
# =========================
DATA_PATH = "data/raw/loan_data.csv"
MODEL_PATH = "models/credit_model.pkl"
RANDOM_STATE = 42


def train():
    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # -------------------------
    # Cleaning + features
    # -------------------------
    df = clean_data(df)
    df = build_features(df)

    # -------------------------
    # Target definition
    # -------------------------
    df["loan_status"] = df["loan_status"].str.lower().str.strip()

    valid_status = ["fully paid", "charged off", "default"]
    df = df[df["loan_status"].isin(valid_status)]

    df["target"] = df["loan_status"].isin(
        ["charged off", "default"]
    ).astype(int)

    # -------------------------
    # Feature selection
    # -------------------------
    features = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "purpose",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
    ]

    X = df[features]
    y = df["target"]

    # -------------------------
    # Columns
    # -------------------------
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # -------------------------
    # Preprocessing pipelines
    # -------------------------
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # -------------------------
    # Model pipeline
    # -------------------------
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                solver="lbfgs"
            )),
        ]
    )

    # -------------------------
    # Train / test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # -------------------------
    # Train
    # -------------------------
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"AUC: {auc:.4f}")

    # -------------------------
    # Save model
    # -------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()
