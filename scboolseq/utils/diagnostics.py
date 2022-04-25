"""
    Tools for assessing the results of scBoolSeq's simulation data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap


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


def expression_profile_scatterplot(
    name: str, frame: pd.DataFrame, group_col: str = "Category", **sns_lmplot_kwargs
):
    """plot the expression profile for a frame
    the frame should be the result of calling

    >>> frame = summarise_by_criteria(data, criteria)
    """
    lmplot_kw = dict(
        x="Mean",
        y="Variance",
        hue=group_col,
        data=frame.sort_values(group_col),
        fit_reg=True,
        scatter_kws={"alpha": 0.4},
    )
    lmplot_kw.update(sns_lmplot_kwargs)
    sns.lmplot(**lmplot_kw)
    plt.title(name)
    plt.ion()
    plt.show()
