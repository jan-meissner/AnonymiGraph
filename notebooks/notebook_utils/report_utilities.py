from typing import Any, Dict, Union

import pandas as pd


def _create_multiindex_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a nested dictionary into a pandas DataFrame with MultiIndex columns.
    """
    df = pd.Series(data).apply(pd.Series).stack().apply(pd.Series).stack().unstack([1, 2])
    df.columns = pd.MultiIndex.from_tuples([(col[0], "" if col[1] == 0 else col[1]) for col in df.columns])
    return df


def _get_color(rank: int, k: int) -> str:
    """
    Returns a color based on the rank.
    """
    return "#{:02x}{:02x}{:02x}{:02x}".format(0, 255, 0, max(0, int((1 - (rank - 1) / k) * 255 * 0.5)))


def _color_by_ranking(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Colors and formats a pandas dataframe created with 'create_multiindex_df' from the output of an Evaluator. For each
    column the k best rows are colored in varying shades of green.
    """
    styled = pd.DataFrame("", index=df.index, columns=df.columns)  # Create an empty DataFrame for styled output

    for col in df.columns.levels[0]:
        if isinstance(df[col], pd.Series):
            ranked_vals = df[col].rank(method="min", ascending=True)
            for rank in range(1, k + 1):
                styled.loc[ranked_vals == rank, col] = f"background-color: {_get_color(rank, k)};"

        elif set(["G", "Ga"]).issubset(df[col].columns):
            abs_diff = (df[col]["G"] - df[col]["Ga"]).abs()
            ranked_diff = abs_diff.rank(method="min", ascending=True)
            for rank in range(1, k + 1):
                styled.loc[ranked_diff == rank, (col, "G")] = f"background-color: {_get_color(rank, k)};"
                styled.loc[ranked_diff == rank, (col, "Ga")] = f"background-color: {_get_color(rank, k)};"

    return styled


def _format_numbers(x: Union[int, float]) -> str:
    return f"{int(x)}" if isinstance(x, float) and x.is_integer() else f"{x:.6f}"


def visualize_evaluator_outputs(data: Dict[str, Any], k: int) -> pd.DataFrame:
    """
    Converts a nested dictionary into a pandas DataFrame with MultiIndex columns. For each
    column the k best rows are colored in varying shades of green.
    """
    df_cleaned = _create_multiindex_df(data)
    styled_df = df_cleaned.style.format(_format_numbers).apply(_color_by_ranking, axis=None, k=k)
    return styled_df
