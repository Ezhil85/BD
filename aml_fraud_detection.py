"""
AML (Anti-Money Laundering) Fraud Detection using Logistic Regression.

Features engineered from typical financial transaction data:
- Transaction amount and velocity
- Customer risk indicators
- Geographic risk (cross-border, high-risk countries)
- Structuring patterns (smurfing detection)
- Unusual timing / frequency
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_aml_dataset(n_samples: int = 10_000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generate a synthetic AML transaction dataset.

    AML red-flag patterns simulated:
    - Structuring / smurfing (amounts just below reporting thresholds)
    - Rapid round-trip transactions
    - High-risk country involvement
    - Unusual transaction hours (late-night)
    - Sudden velocity spikes
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def _legit_records(n):
        return {
            "transaction_amount":       np.random.lognormal(mean=6.5, sigma=1.2, size=n),
            "num_transactions_24h":     np.random.poisson(lam=3, size=n),
            "num_transactions_7d":      np.random.poisson(lam=15, size=n),
            "avg_transaction_amount":   np.random.lognormal(mean=6.5, sigma=0.8, size=n),
            "transaction_hour":         np.random.randint(7, 22, size=n),
            "is_cross_border":          np.random.binomial(1, p=0.10, size=n),
            "high_risk_country":        np.random.binomial(1, p=0.05, size=n),
            "is_cash":                  np.random.binomial(1, p=0.15, size=n),
            "account_age_days":         np.random.randint(30, 3650, size=n),
            "days_since_last_txn":      np.random.exponential(scale=3, size=n),
            "num_counterparties_30d":   np.random.poisson(lam=4, size=n),
            "amount_deviation_ratio":   np.abs(np.random.normal(0, 0.5, size=n)),
            "round_amount_flag":        np.random.binomial(1, p=0.08, size=n),
            "previously_flagged":       np.random.binomial(1, p=0.02, size=n),
            "customer_risk_score":      np.random.uniform(0, 0.4, size=n),
            "label": np.zeros(n, dtype=int),
        }

    def _fraud_records(n):
        # Structuring: amounts cluster just below 10 000 reporting threshold
        structuring_amounts = np.random.uniform(8_500, 9_999, size=n)
        # Mix with occasional large amounts
        large_amounts = np.random.lognormal(mean=9, sigma=0.5, size=n)
        mask = np.random.binomial(1, p=0.6, size=n).astype(bool)
        amounts = np.where(mask, structuring_amounts, large_amounts)

        return {
            "transaction_amount":       amounts,
            "num_transactions_24h":     np.random.poisson(lam=12, size=n),
            "num_transactions_7d":      np.random.poisson(lam=50, size=n),
            "avg_transaction_amount":   np.random.lognormal(mean=8, sigma=0.8, size=n),
            "transaction_hour":         np.random.choice(
                                            list(range(0, 6)) + list(range(22, 24)), size=n
                                        ),
            "is_cross_border":          np.random.binomial(1, p=0.65, size=n),
            "high_risk_country":        np.random.binomial(1, p=0.55, size=n),
            "is_cash":                  np.random.binomial(1, p=0.60, size=n),
            "account_age_days":         np.random.randint(1, 90, size=n),
            "days_since_last_txn":      np.random.exponential(scale=0.3, size=n),
            "num_counterparties_30d":   np.random.poisson(lam=20, size=n),
            "amount_deviation_ratio":   np.abs(np.random.normal(2.5, 1.0, size=n)),
            "round_amount_flag":        np.random.binomial(1, p=0.45, size=n),
            "previously_flagged":       np.random.binomial(1, p=0.30, size=n),
            "customer_risk_score":      np.random.uniform(0.5, 1.0, size=n),
            "label": np.ones(n, dtype=int),
        }

    df = pd.concat(
        [pd.DataFrame(_legit_records(n_legit)), pd.DataFrame(_fraud_records(n_fraud))],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Velocity ratio: recent burst vs baseline
    df["velocity_ratio"] = df["num_transactions_24h"] / (
        df["num_transactions_7d"] / 7 + 1e-6
    )

    # Structuring flag: amount within 5 % below common reporting thresholds
    thresholds = [10_000, 50_000, 100_000]
    df["structuring_flag"] = 0
    for thr in thresholds:
        lower = thr * 0.95
        df["structuring_flag"] = np.where(
            (df["transaction_amount"] >= lower) & (df["transaction_amount"] < thr),
            1,
            df["structuring_flag"],
        )

    # Off-hours flag (midnight – 6 am)
    df["off_hours_flag"] = (df["transaction_hour"] < 6).astype(int)

    # Log-transform skewed numerics
    df["log_amount"]          = np.log1p(df["transaction_amount"])
    df["log_avg_amount"]      = np.log1p(df["avg_transaction_amount"])
    df["log_account_age"]     = np.log1p(df["account_age_days"])
    df["log_days_since_last"] = np.log1p(df["days_since_last_txn"])

    # Composite risk: geographic + cash + previously flagged
    df["composite_risk"] = (
        df["is_cross_border"]
        + df["high_risk_country"]
        + df["is_cash"]
        + df["previously_flagged"]
        + df["customer_risk_score"]
    )

    return df


FEATURE_COLS = [
    "log_amount",
    "log_avg_amount",
    "num_transactions_24h",
    "num_transactions_7d",
    "velocity_ratio",
    "transaction_hour",
    "off_hours_flag",
    "is_cross_border",
    "high_risk_country",
    "is_cash",
    "log_account_age",
    "log_days_since_last",
    "num_counterparties_30d",
    "amount_deviation_ratio",
    "round_amount_flag",
    "previously_flagged",
    "customer_risk_score",
    "structuring_flag",
    "composite_risk",
]


# ---------------------------------------------------------------------------
# 3. Handle class imbalance via oversampling
# ---------------------------------------------------------------------------

def balance_classes(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    X_combined = pd.concat([X_train, y_train.rename("label")], axis=1)
    majority = X_combined[X_combined["label"] == 0]
    minority = X_combined[X_combined["label"] == 1]
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE
    )
    balanced = pd.concat([majority, minority_upsampled]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return balanced.drop("label", axis=1), balanced["label"]


# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            C=0.5,
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# 5. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("AML FRAUD DETECTION — MODEL EVALUATION")
    print("=" * 60)
    print(f"\nROC-AUC Score      : {roc_auc:.4f}")
    print(f"Avg Precision Score: {avg_prec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legit", "Actual Fraud"],
        columns=["Pred Legit", "Pred Fraud"],
    )
    print(cm_df)

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
    }


def cross_validate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n5-Fold CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")


# ---------------------------------------------------------------------------
# 6. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(pipeline: Pipeline, feature_names: list) -> None:
    coef = pipeline.named_steps["clf"].coef_[0]
    importance = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh", color="steelblue")
    plt.title("Logistic Regression Feature Importance (|coefficient|)")
    plt.xlabel("|Coefficient|")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("\nFeature importance chart saved → feature_importance.png")


def plot_roc_pr_curves(y_test, y_proba) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(rec, prec, color="green", lw=2, label=f"AP = {ap:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", dpi=150)
    plt.close()
    print("ROC / PR curves saved → roc_pr_curves.png")


# ---------------------------------------------------------------------------
# 7. Prediction interface
# ---------------------------------------------------------------------------

def predict_transaction(pipeline: Pipeline, transaction: dict, threshold: float = 0.5) -> dict:
    """
    Predict whether a single transaction is AML fraud.

    Parameters
    ----------
    pipeline    : trained sklearn Pipeline
    transaction : dict with raw transaction fields
    threshold   : classification cutoff (lower = more sensitive)

    Returns
    -------
    dict with fraud probability, label, and risk level
    """
    df_raw = pd.DataFrame([transaction])
    df_feat = engineer_features(df_raw)

    # Fill any missing engineered columns with 0
    for col in FEATURE_COLS:
        if col not in df_feat.columns:
            df_feat[col] = 0

    prob  = pipeline.predict_proba(df_feat[FEATURE_COLS])[0, 1]
    label = int(prob >= threshold)

    if prob >= 0.75:
        risk_level = "HIGH"
    elif prob >= 0.50:
        risk_level = "MEDIUM"
    elif prob >= 0.25:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(label),
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic AML transaction dataset …")
    df_raw = generate_aml_dataset(n_samples=10_000, fraud_ratio=0.05)
    df     = engineer_features(df_raw)

    print(f"Dataset shape : {df.shape}")
    print(f"Fraud rate    : {df['label'].mean():.2%}")

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print("\nBalancing training classes via oversampling …")
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    print("Training Logistic Regression …")
    pipeline = train_model(X_train_bal, y_train_bal)

    results = evaluate_model(pipeline, X_test, y_test)

    cross_validate_model(pipeline, X, y)

    plot_feature_importance(pipeline, FEATURE_COLS)
    plot_roc_pr_curves(y_test, results["y_proba"])

    # --- Demo: score two example transactions ---
    sample_legit = {
        "transaction_amount": 1_200,
        "num_transactions_24h": 2,
        "num_transactions_7d": 10,
        "avg_transaction_amount": 950,
        "transaction_hour": 14,
        "is_cross_border": 0,
        "high_risk_country": 0,
        "is_cash": 0,
        "account_age_days": 730,
        "days_since_last_txn": 3,
        "num_counterparties_30d": 3,
        "amount_deviation_ratio": 0.3,
        "round_amount_flag": 0,
        "previously_flagged": 0,
        "customer_risk_score": 0.1,
    }

    sample_suspicious = {
        "transaction_amount": 9_850,   # structuring — just below 10 000
        "num_transactions_24h": 18,
        "num_transactions_7d": 72,
        "avg_transaction_amount": 9_500,
        "transaction_hour": 2,         # late night
        "is_cross_border": 1,
        "high_risk_country": 1,
        "is_cash": 1,
        "account_age_days": 12,
        "days_since_last_txn": 0.1,
        "num_counterparties_30d": 25,
        "amount_deviation_ratio": 3.2,
        "round_amount_flag": 1,
        "previously_flagged": 1,
        "customer_risk_score": 0.85,
    }

    print("\n" + "=" * 60)
    print("REAL-TIME TRANSACTION SCORING DEMO")
    print("=" * 60)
    for label, txn in [("Legitimate sample", sample_legit), ("Suspicious sample", sample_suspicious)]:
        result = predict_transaction(pipeline, txn)
        print(f"\n{label}:")
        print(f"  Fraud probability : {result['fraud_probability']:.4f}")
        print(f"  Prediction        : {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
        print(f"  Risk level        : {result['risk_level']}")


if __name__ == "__main__":
    main()
