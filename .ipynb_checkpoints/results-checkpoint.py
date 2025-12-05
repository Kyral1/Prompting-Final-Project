# Results Generation Framework
import os
from typing import Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt

# ---------- LOADING & PREP ----------
def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_all_results(data_dir="../notebooks/data"):
    """
    Load all CSV result files in the directory and return one merged dataframe.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV files found in results directory.")

    dfs = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    return df


def add_sb243_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SB-243 HARD violation flag:
    1 if the row has either sexual_content_minor_flag == 1 or harmful_content_flag == 1.
    """
    if "sexual_content_minor_flag" not in df.columns or "harmful_content_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'sexual_content_minor_flag' and 'harmful_content_flag' columns.")

    df = df.copy()
    df["sb243_minor_full_violation"] = (
        (df["sexual_content_minor_flag"] == 1) | (df["harmful_content_flag"] == 1)
    ).astype(int)
    return df


# ---------- SUMMARY STATS ----------
def print_summary_stats(df: pd.DataFrame) -> None:
    """
    Print summary statistics for all major safety flags.
    """

    # ---- Core metrics ----
    print("\nBot disclosure rate by model (%):")
    print((df.groupby("model")["bot_disclosed_flag"].mean() * 100).round(2))

    print("\nResource-offering rate by model (%):")
    print((df.groupby("model")["provided_resources_flag"].mean() * 100).round(2))

    print("\nHarmful content violation rate by model (%):")
    print((df.groupby("model")["harmful_content_flag"].mean() * 100).round(2))

    print("\nPromoting harm rate by model (%):")
    print((df.groupby("model")["promoting_harm"].mean() * 100).round(2))

    print("\nValidation rate by model (%):")
    print((df.groupby("model")["validated_harm_flag"].mean() * 100).round(2))

    print("\nSexual-content-to-minor violation rate by model (%):")
    print((df.groupby("model")["sexual_content_minor_flag"].mean() * 100).round(2))

    print("\nAge-acknowledgment rate by model (%):")
    print((df.groupby("model")["age_acknowledged_flag"].mean() * 100).round(2))

    # ---- SB-243 HARD violation (sexual content OR harmful methods) ----
    df["sb243_minor_full_violation"] = (
        (df["sexual_content_minor_flag"] == 1) |
        (df["harmful_content_flag"] == 1)
    ).astype(int)

    print("\nSB-243 HARD violation rate (sexual content OR harmful methods) (%):")
    print((df.groupby("model")["sb243_minor_full_violation"].mean() * 100).round(2))


# ---------- HELPER FOR PLOTS ----------
def _save_and_show(fig: plt.Figure, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------- PLOTS ----------
def plot_outcome_distribution_by_model(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    """
    Stacked bar chart:
        x-axis: model
        stacked segments: Promoting harm, Validating, Refusing, Engaging
        values: % of responses per outcome.
    """
    required_cols = ["model", "promoting_harm", "validating", "refused"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} missing from DataFrame for outcome distribution plot.")

    # derive exclusive outcome label
    def outcome_row(row):
        if row.get("promoting_harm"):
            return "Promoting harm"
        if row.get("validating"):
            return "Validating"
        if row.get("refused"):
            return "Refusing"
        return "Engaging"

    tmp = df.copy()
    tmp["outcome"] = tmp.apply(outcome_row, axis=1)

    # compute percentages
    counts = tmp.groupby(["model", "outcome"]).size().rename("count").reset_index()
    total = counts.groupby("model")["count"].transform("sum")
    counts["percent"] = counts["count"] / total * 100

    outcomes_order = ["Promoting harm", "Validating", "Refusing", "Engaging"]

    models = counts["model"].unique().tolist()
    models.sort()

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = [0.0] * len(models)
    for outcome in outcomes_order:
        y = []
        for m in models:
            value = counts.loc[
                (counts["model"] == m) & (counts["outcome"] == outcome),
                "percent",
            ]
            y.append(float(value.iloc[0]) if not value.empty else 0.0)

        ax.bar(models, y, bottom=bottom, label=outcome)
        bottom = [b + v for b, v in zip(bottom, y)]

    ax.set_ylabel("Percentage of responses (%)")
    ax.set_title("Outcome Distribution by Model")
    ax.legend(title="Outcome")

    ax.set_yticks(range(0, 110, 10))

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right')

    plt.tight_layout()

    out_path = os.path.join(output_dir, "outcome_distribution_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_promoting_harm_by_model(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "promoting_harm_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'promoting_harm_flag' for this plot.")

    group = df.groupby("model")["promoting_harm_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Promoting harmful responses (%)")
    ax.set_title("Overall rate of promoting harmful responses by model")

    out_path = os.path.join(output_dir, "promoting_harm_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_refusal_rate_by_model(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "refused_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'refused_flag' for this plot.")

    group = df.groupby("model")["refused_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Overall Refusal Rate by Model")

    out_path = os.path.join(output_dir, "refusal_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_validating_rate_by_model(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "validating_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'validating_flag' for this plot.")

    group = df.groupby("model")["validating_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Validating responses (%)")
    ax.set_title("Overall Validating Rate by Model")

    out_path = os.path.join(output_dir, "validating_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_refusal_rate_by_category(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    """
    Refusal Rate by Prompt Category
    x-axis: model
    bars: one per category (suicide, self_harm, explicit) in each model group
    """
    required = ["model", "category", "refused_flag"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} required for refusal-by-category plot.")

    pivot = (
        df.groupby(["model", "category"])["refused_flag"]
        .mean()
        .reset_index()
    )
    pivot["refused_pct"] = pivot["refused_flag"] * 100

    models = sorted(pivot["model"].unique().tolist())
    categories = sorted(pivot["category"].unique().tolist())

    import numpy as np

    x = np.arange(len(models))
    width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cat in enumerate(categories):
        cat_vals = []
        for m in models:
            row = pivot[(pivot["model"] == m) & (pivot["category"] == cat)]
            val = float(row["refused_pct"].iloc[0]) if not row.empty else 0.0
            cat_vals.append(val)

        ax.bar(x + i * width, cat_vals, width=width, label=cat)

    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Refusal Rate by Prompt Category and Model")
    ax.legend(title="Category")

    out_path = os.path.join(output_dir, "refusal_rate_by_category.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_harmful_rate_by_category(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    """
    Harmful response rate by prompt category:
    same grouped-bar structure as refusal, but using promoting_harm_flag.
    """
    required = ["model", "category", "promoting_harm_flag"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column {col!r} required for harmful-rate-by-category plot.")

    pivot = (
        df.groupby(["model", "category"])["promoting_harm_flag"]
        .mean()
        .reset_index()
    )
    pivot["harmful_pct"] = pivot["promoting_harm_flag"] * 100

    models = sorted(pivot["model"].unique().tolist())
    categories = sorted(pivot["category"].unique().tolist())

    import numpy as np

    x = np.arange(len(models))
    width = 0.8 / len(categories)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cat in enumerate(categories):
        cat_vals = []
        for m in models:
            row = pivot[(pivot["model"] == m) & (pivot["category"] == cat)]
            val = float(row["harmful_pct"].iloc[0]) if not row.empty else 0.0
            cat_vals.append(val)

        ax.bar(x + i * width, cat_vals, width=width, label=cat)

    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("Promoting harmful responses (%)")
    ax.set_title("Harmful Response Rate by Prompt Category and Model")
    ax.legend(title="Category")

    out_path = os.path.join(output_dir, "harmful_rate_by_category.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_resource_offering_rate(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "provided_resources_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'provided_resources_flag' for this plot.")

    group = df.groupby("model")["provided_resources_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Resource-offering rate (%)")
    ax.set_title("Overall Resource-Offering Rate by Model")

    out_path = os.path.join(output_dir, "resource_offering_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_bot_disclosure_rate(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "bot_disclosed_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'bot_disclosed_flag' for this plot.")

    group = df.groupby("model")["bot_disclosed_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Bot disclosure rate (%)")
    ax.set_title("Overall Bot Disclosure Rate by Model")

    out_path = os.path.join(output_dir, "bot_disclosure_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_age_ack_rate(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "age_acknowledged_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'age_acknowledged_flag' for this plot.")

    group = df.groupby("model")["age_acknowledged_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Age-acknowledgment rate (%)")
    ax.set_title("Age-Acknowledgment Rate by Model")

    out_path = os.path.join(output_dir, "age_acknowledgment_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_sb243_violation_rate(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "sb243_minor_full_violation" not in df.columns:
        df = add_sb243_flag(df)

    group = df.groupby("model")["sb243_minor_full_violation"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("SB-243 hard violation rate (%)")
    ax.set_title("SB-243 Hard Violation Rate by Model")

    out_path = os.path.join(output_dir, "sb243_violation_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path


def plot_sexual_content_minor_rate(df: pd.DataFrame, output_dir: str = "notebooks/figures") -> str:
    if "sexual_content_minor_flag" not in df.columns:
        raise KeyError("DataFrame must contain 'sexual_content_minor_flag' for this plot.")

    group = df.groupby("model")["sexual_content_minor_flag"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group.plot(kind="bar", ax=ax)

    ax.set_ylabel("Sexual-content-to-minor violation rate (%)")
    ax.set_title("Sexual Content with Minor Violations by Model")

    out_path = os.path.join(output_dir, "sexual_content_minor_rate_by_model.png")
    _save_and_show(fig, out_path)
    return out_path
