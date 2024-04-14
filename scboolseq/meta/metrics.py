""" """

import pandas as pd


def meta_marker_counter(meta_bin: pd.DataFrame) -> pd.DataFrame:
    """ """
    meta_cols = meta_bin.columns
    meta_index = meta_bin.index
    _by_group_markers = {}
    _by_group_negative_markers = {}

    for _group, _gi in meta_bin.groupby(meta_bin.index).groups.items():
        _gno_na = meta_bin.loc[_group, :].dropna()
        _active_in_group = _gno_na.index[_gno_na == 1]
        _inactive_in_group = _gno_na.index[_gno_na == 0]
        _active_elsewhere = meta_cols[
            meta_bin.loc[meta_index.difference(_gi), :].any() == 1
        ]
        _inactive_elsewhere = meta_cols[
            meta_bin.loc[meta_index.difference(_gi), :].all() == 0
        ]
        _by_group_markers[_group] = _active_in_group.difference(_active_elsewhere)
        _by_group_negative_markers[_group] = _inactive_in_group.difference(
            _inactive_elsewhere
        )

    return pd.DataFrame(
        {
            "n_positive_markers": {g: len(v) for g, v in _by_group_markers.items()},
            "n_negative_markers": {
                g: len(v) for g, v in _by_group_negative_markers.items()
            },
        }
    )
