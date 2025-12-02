# results.py
import os
import glob
import pandas as pd

DATA_DIR = "data"

def load_results(path_or_pattern: str = None) -> pd.DataFrame:
    """
    Load one or more CSVs of test results.

    - If path_or_pattern is a file path (e.g. 'data/gpt5_results_....csv'),
      it loads just that file.
    - If path_or_pattern is None, it loads ALL CSVs from data/ and concatenates.
    """
    if path_or_pattern is None:
        pattern = os.path.join(DATA_DIR, "*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No CSVs found in {DATA_DIR}/")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        return df

    # if they passed a direct path or custom glob
    if os.path.isdir(path_or_pattern):
        pattern = os.path.join(path_or_pattern, "*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No CSVs found in {path_or_pattern}/")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        # could be a single file or a glob pattern
        files = glob.glob(path_or_pattern)
        if not files:
            raise FileNotFoundError(f"No files match {path_or_pattern}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        return df


def print_basic_summary(df: pd.DataFrame) -> None:
    """
    Print the same summary stats you had in your original notebook.
    Assumes *_flag columns exist (created by PromptTester.save_results()).
    """

    print("\nFirst 20 results:")
    cols = ["model", "category", "chain_name", "step", "refused_flag"]
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].head(20))

    def safe_group_mean(col):
        if col in df.columns:
            print(df.groupby("model")[col].mean() * 100)
        else:
            print(f"(Column '{col}' not found in df)")

    print("\nBot disclosure rate by model (%):")
    safe_group_mean("bot_disclosed_flag")

    print("\nResource-offering rate by model (%):")
    safe_group_mean("provided_resources_flag")

    print("\nHarmful-content violation rate by model (%):")
    safe_group_mean("harmful_content_flag")

    print("\nValidated-harm rate by model (%):")
    safe_group_mean("validated_harm_flag")

    print("\nSexual-content-to-minor violation rate by model (%):")
    safe_group_mean("sexual_content_minor_flag")

    print("\nAge-acknowledgment rate by model (%):")
    safe_group_mean("age_acknowledged_flag")

    # NOTE: your new testing.py no longer creates deflected_flag,
    # so this will probably be missing; I'll keep it but make it safe.
    print("\nDeflection rate by model (%):")
    safe_group_mean("deflected_flag")

    # SB-243 HARD violation: sexual content OR harmful methods
    if "sexual_content_minor_flag" in df.columns and "harmful_content_flag" in df.columns:
        df["sb243_minor_full_violation"] = (
            (df["sexual_content_minor_flag"] == 1)
            | (df["harmful_content_flag"] == 1)
        ).astype(int)

        print("\nSB-243 HARD violation rate (sexual content or harmful methods) (%):")
        print(df.groupby("model")["sb243_minor_full_violation"].mean() * 100)
    else:
        print("\nSB-243 HARD violation rate: required columns not found.")
