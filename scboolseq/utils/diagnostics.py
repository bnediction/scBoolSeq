"""
    Tools for assessing the results of scBoolSeq's simulation data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotnine as ggplot


def summarise_by_criteria(data: pd.DataFrame, criteria: pd.DataFrame) -> pd.DataFrame:
    """Summarise by criteria to get the needed mean vs std plots"""
    return pd.DataFrame(
        {"Mean": data.mean(), "Variance": data.var(), "Category": criteria.Category}
    )


def compare_profiles(
    criteria: pd.DataFrame,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str = "frame_a",
    label_b: str = "frame_b",
):
    """Compare the profile of two samples sharing a single criteria DataFrame"""
    summary_a = summarise_by_criteria(frame_a, criteria)
    summary_b = summarise_by_criteria(frame_b, criteria)
    summary_a["Data"] = label_a
    summary_b["Data"] = label_b
    merged = pd.concat([summary_a, summary_b], axis="rows")
    merged = merged.reset_index()

    return merged
    
