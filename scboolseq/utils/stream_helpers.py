"""
    Temporarily storing helper functions for branching event
    reconstruction.
"""

from typing import Iterator, Dict, Tuple, Union, List, Optional, Iterable
from itertools import chain, combinations  # , product  # For powerset calculation
from functools import partial

# ^in order to pre-load functions before passing them to
#  pd.DataFrame.index.map()

import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard as jaccard_distance

import networkx as nx

from ..simulation import biased_simulation_from_binary_state

RandomWalkGenerator = Union[Iterator[Dict[str, int]], List[Dict[str, int]]]
SampleCountSpec = Union[int, range, Iterable[int]]


def state_to_str(state: Dict[str, int]) -> str:
    """Convert a state to a string"""
    return "".join(str(value) for value in state.values())


def graph_node_to_dict(async_dynamics, node: str):
    """Convert a node of the transition graph to a dictionary.
    This function takes the transition graph calculated via
    colomoto.minibn.FullyAsynchronousDynamics.partial_dynamics(initial_state)
    """
    return dict(zip(async_dynamics.nodes, map(int, list(node))))


def find_terminal_nodes_indexes(digraph: nx.DiGraph):
    """Find terminal nodes of a netorkx.DiGraph (directed graph)"""
    return [i for i in digraph.nodes if len(list(digraph.successors(i))) == 0]


def condensation_node_to_dicts(async_dynamics, condensation, node: int):
    """Return a list with all the condensation nodes formatted
    as dictionaries."""
    return [
        dict(zip(async_dynamics.nodes, w)) for w in condensation.nodes[node]["members"]
    ]


def condensation_node_to_str(node) -> str:
    """Convert a netorkx condenstation node to a string"""
    return "".join(w for w in node["members"])


def powerset(iterable):
    """
    Generate all possible combinations
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    iter_ls = list(iterable)
    return chain.from_iterable(
        combinations(iter_ls, r) for r in range(len(iter_ls) + 1)
    )


def shortest_path_passing_by(graph, start, visit, end) -> List[str]:
    """Given a graph, a starting node, an end node, compute the
    shortest paths :
        (start, visit)
        (visit, end)

    And merge them as follows :
    start -> ... -> visit -> ... -> end
    """
    format_node = lambda x: x if isinstance(x, str) else state_to_str(x)

    _fist_path = nx.shortest_path(graph, format_node(start), format_node(visit))
    _second_path = nx.shortest_path(graph, format_node(visit), format_node(end))

    if not _fist_path[-1] == _second_path[0]:
        raise ValueError(
            f"`{_fist_path[-1]}` != `{_second_path[0]}`, inconsistent path"
        )

    _ = _fist_path.pop()

    return _fist_path + _second_path


def boolean_df_query_generator(state: Dict[str, int]):
    """Generate a query to get the dataframe row(s)
    matching the given boolean state.

    example:
        >>> state = {"gene1": 1, "gene2": 0}
        >>> query = boolean_df_query_generator(state)
        >>> trajectory_df.query(query)
    """
    return " and ".join(f"{key} == {value}" for key, value in state.items())


def jaccard_similarity(u, v, w=None) -> float:
    """Compute the Jaccard similarity index
    defined as the cardinal of the intersection
    divided by the cardinal of the union.

    returns : 1.0 - scipy.spatial.distance.jaccard(u, v, w)
    """
    return 1.0 - jaccard_distance(u, v, w)


def trajectory_to_data_frame(
    random_walk_generator: RandomWalkGenerator,
) -> pd.DataFrame:
    """Build and return a DataFrame from a random walk.

    A random walk generator is obtained as follows (minimal example) :
    >>> bn: colomoto.minibn.BooleanNetwork
    >>> dyn = colomoto.minibn.FullyAsynchronousDynamics(bn)
    >>> initial_state: Dict[str,int] = {'gene1': 0, 'gene2': 1}
    >>> n_steps = 100
    >>> _random_walk = dyn.random_walk(initial_state, steps=n_steps)

    The `_random_walk` objet should be passed to this function.

    Returns:
        A DataFrame containing the genes (variables) as columns and one row
        for each observed state in the random walk. The index is simply
        the ordered set of integers in the range [0, n_steps).

    """
    _rnd_wlks = list(random_walk_generator)
    _rnd_walk_df = pd.concat(
        [pd.DataFrame(walk, index=[i]) for i, walk in enumerate(_rnd_wlks)], axis="rows"
    )
    _rnd_walk_df.index.name = "step"
    return _rnd_walk_df


def merge_random_walks(
    walk_a: pd.DataFrame,
    walk_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    branching_point_query: str,
    attractor_a_query: str,
    attractor_b_query: str,
    distinguish_attractors: bool = True,
    confound_observations: bool = True,
) -> pd.DataFrame:
    """

    Merge random walks in order to simulate a bifurcation process.

    Parameters
    ----------
        walk_a : A trajectory ending in a fixed-point attractor.
        walk_b : Idem, but on a different attractor.

            note : These two should have at least one shared point


    TODO : allow params [
        branching_point_query,
        attractor_[1,2]_query
    ] to be either strings or dictionnaries
    """
    # deep-copy our frames to avoid overwritting them
    _walk_a = walk_a.copy(deep=True)
    _walk_b = walk_b.copy(deep=True)

    def trajectory_tagger(idx, branching_point, attractor, attractor_tag):
        """Tag a bifurcation process. Use a single branching point as
        reference
            Tags include:
                "common" <=> before the branching point
                "split"  <=> the branching point
                "branch" <=> states after the branching point and before the attractor
                "attractor" <=> the attractor
        """
        if idx < branching_point:
            return f"common_{idx}"
        elif idx == branching_point:
            return f"split_{idx}"
        elif idx == attractor:
            return (
                f"attractor_{idx}_{attractor_tag}"
                if distinguish_attractors
                else f"attractor_{idx}"
            )
        elif idx > branching_point:
            return f"branch_{idx}"
        else:
            raise ValueError(
                f"Undecidable index {idx}, bp: {branching_point}, attr: {attractor}"
            )

    # Tag the first walk's index, according to the branching point and attractor queries
    _walk_a = _walk_a.set_index(
        _walk_a.index.map(
            lambda x: trajectory_tagger(
                idx=x,
                branching_point=_walk_a.query(branching_point_query).index[
                    0
                ],  # safe, only one branching point
                attractor=_walk_a.query(attractor_a_query).index[
                    0
                ],  # safe, only one attractor
                attractor_tag=label_a,
            )
        )
    )

    # Tag the second walk's index, according to the branching point and attractor queries
    _walk_b = _walk_b.set_index(
        _walk_b.index.map(
            lambda x: trajectory_tagger(
                idx=x,
                branching_point=_walk_b.query(branching_point_query).index[
                    0
                ],  # idem as for _walk_b
                attractor=_walk_b.query(attractor_b_query).index[0],
                attractor_tag=label_b,
            )
        )
    )

    f_walk_index_formatter = lambda frame, label: frame.set_index(
        frame.index.astype(str).map(lambda x: f"{label}_{x}")
    )

    _walk_a = f_walk_index_formatter(_walk_a, label_a)
    _walk_b = f_walk_index_formatter(_walk_b, label_b)

    def index_masker(unique_idx, distinguish_attractors: bool = True):
        """Confound uniquely labelled samples within a trajectory."""
        if "common" in unique_idx:
            return "common"
        elif "split" in unique_idx:
            return "split"
        elif "branch" in unique_idx:
            return "branch"
        elif "attractor" in unique_idx:
            return unique_idx if distinguish_attractors else "attractor"
        else:
            raise ValueError(f"Unknown tag `{unique_idx}` found on index")

    _result = pd.concat([_walk_a, _walk_b], axis="rows")

    _partial_index_masker = partial(
        index_masker, distinguish_attractors=distinguish_attractors
    )
    if confound_observations:
        _result = _result.set_index(_result.index.map(_partial_index_masker))

    return _result


def merge_binary_trajectories(
    trajectories: List[pd.DataFrame],
    labels: List[str],
    branching_point_query: str,
    attractor_queries: List[str],
    distinguish_attractors: bool = True,
    confound_observations: bool = True,
    df_index_name: Optional[str] = None,
) -> pd.DataFrame:
    """

    Merge random walks in order to simulate a multifurcation process.

    Only one branching point is expected to exist,
    leading to multiple attractors.

    Parameters
    ----------
        walk_a : A trajectory ending in a fixed-point attractor.
        walk_b : Idem, but on a different attractor.

            note : These two should have at least one shared point


    TODO : allow params [
        branching_point_query,
        attractor_[1,2]_query
    ] to be either strings or dictionnaries
    """
    # Integrity check
    if not len(trajectories) == len(labels) == len(attractor_queries):
        _len_err_ls = [
            "Lengths of entry lists should all be equal.",
            f"len(trajectories) = {len(trajectories)} !=",
            f"len(labels) = {len(labels)} !=",
            f"len(attractor_queries) = {len(attractor_queries)}",
        ]
        raise ValueError(" ".join(_len_err_ls))

    # deep-copy our frames to avoid overwritting them
    _trajectories = [traj.copy(deep=True) for traj in trajectories]

    def trajectory_tagger(idx, branching_point, attractor, attractor_tag):
        """Tag a bifurcation process.
        Use a single branching point as reference
            Tags include:
                "common" <=> before the branching point
                "split"  <=> the branching point
                "branch" <=> states after the branching point and before the attractor
                "attractor" <=> the attractor
        """
        if idx < branching_point:
            return f"common_{idx}"
        elif idx == branching_point:
            return f"split_{idx}"
        elif idx == attractor:
            return (
                f"attractor_{attractor_tag}_{idx}"
                if distinguish_attractors
                else f"attractor_{idx}"
            )
        elif idx > branching_point:
            return (
                f"branch_{attractor_tag}_{idx}"
                if distinguish_attractors
                else f"branch_{idx}"
            )
        else:
            raise ValueError(
                f"Undecidable index {idx}, bp: {branching_point}, attr: {attractor}"
            )

    trajectory_index_tagger = lambda _traj, _attr_query, _attr_label: _traj.set_index(
        _traj.index.map(
            lambda x: trajectory_tagger(
                idx=x,
                branching_point=_traj.query(branching_point_query).index[
                    0
                ],  # safe, only one branching point
                attractor=_traj.query(_attr_query).index[
                    0
                ],  # safe, only one attractor per trajectory
                attractor_tag=_attr_label,
            )
        )
    )

    # Tag each of the trajectories indexes, according to the branching point
    # and their respective attractor queries
    _trajectories = list(
        map(trajectory_index_tagger, trajectories, attractor_queries, labels)
    )

    # f_walk_index_formatter = lambda frame, label: frame.set_index(
    #    frame.index.astype(str).map(lambda x: f"{label}_{x}")
    # )

    # _trajectories = list(map(f_walk_index_formatter, _trajectories, labels))

    def index_masker(unique_idx, distinguish_attractors: bool = True):
        """Confound uniquely labelled samples within a trajectory."""
        if "common" in unique_idx:
            return "common"
        elif "split" in unique_idx:
            return "split"
        elif "branch" in unique_idx:
            return (
                "_".join(unique_idx.split("_")[:-1])
                if distinguish_attractors
                else "branch"
            )
        elif "attractor" in unique_idx:
            return (
                "_".join(unique_idx.split("_")[:-1])
                if distinguish_attractors
                else "attractor"
            )
        else:
            raise ValueError(f"Unknown tag `{unique_idx}` found on index")

    _result = pd.concat(_trajectories, axis="rows")

    _partial_index_masker = partial(
        index_masker, distinguish_attractors=distinguish_attractors
    )
    if confound_observations:
        _result = _result.set_index(_result.index.map(_partial_index_masker))

    _result.index.name = df_index_name or "label"

    return _result


# TODO : move this function to scBoolSeq.dynamics
def simulate_from_boolean_trajectory(
    boolean_trajectory_df: pd.DataFrame,
    criteria_df: pd.DataFrame,
    n_samples_per_state: SampleCountSpec = 300,
    rng_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate `n_samples_per_state`, for each one of the
    states found in `boolean_trajectory_df`.
    The biased simulation from the binary state is performed
    according to `criteria_df`.

    Parameter `n_samples_per_state` is of type SampleCountSpec,
    defined as :
    >>> SampleCountSpec = Union[int, range, Iterable[int]]
    The behaviour changes depending on the type.
    * If an int is given, all states of the boolean trajectory will
      be sampled `n_samples_per_state` times.
    * If a range is given, a random number of samples (within the range)
      will be created for each observation.
    * If a list is given


    If specified, parameter `rng_seed` allows 100% reproductible results,
    which means that given the same set of parameters with the given seed
    will always produce the same pseudo random values.

    The aim of this function is generating synthetic data to be
    used as an input to STREAM, in order to evaluate the performance
    of the PROFILE-based simulation method we have developped.

    Returns
    -------
        A tuple : (simulated_expression_dataframe, metadata)

    """
    # for all runs to obtain the same results the seeds of each run should be fixed
    _rng = np.random.default_rng(rng_seed)
    _simulation_seeds = _rng.integers(
        123, rng_seed, size=len(boolean_trajectory_df.index)
    )
    _n_states = len(boolean_trajectory_df.index)

    # multiple dispatch for n_samples_per_state
    if isinstance(n_samples_per_state, int):
        sample_sizes = [n_samples_per_state] * _n_states
    elif isinstance(n_samples_per_state, range):
        sample_sizes = _rng.integers(
            n_samples_per_state.start,
            n_samples_per_state.stop,
            size=len(boolean_trajectory_df.index),
        )
    elif isinstance(n_samples_per_state, Iterable):
        sample_sizes = list(n_samples_per_state)
        # check we have enough sample sizes for each one of the observed states
        if not len(sample_sizes) == _n_states:
            raise ValueError(
                " ".join(
                    [
                        "`n_samples_per_state` should contain",
                        f"exactly {_n_states} entries, received {len(sample_sizes)}",
                    ]
                )
            )
    else:
        raise TypeError(
            f"Invalid type `{type(n_samples_per_state)}` for parameter n_samples_per_state"
        )

    # generate synthetic samples
    synthetic_samples = []
    for size, rnd_walk_step, _rng_seed in zip(
        sample_sizes, boolean_trajectory_df.iterrows(), _simulation_seeds
    ):
        _idx, binary_state = rnd_walk_step

        synthetic_samples.append(
            biased_simulation_from_binary_state(
                binary_state.to_frame().T,
                criteria_df,
                n_samples=size,
                seed=_rng_seed,
            )
            .reset_index()
            .rename(columns={"index": "kind"})
        )

    # merge all experiments into a single frame
    synthetic_single_cell_experiment = pd.concat(
        synthetic_samples, axis="rows", ignore_index=True
    )

    # create an informative, artificial, and unique index
    synthetic_single_cell_experiment = synthetic_single_cell_experiment.set_index(
        synthetic_single_cell_experiment.kind
        + "_"
        + synthetic_single_cell_experiment.reset_index().index.map(
            lambda x: f"obs{str(x)}"
        )
    )

    # Create a colour map for different cell types
    _RGB_values = list("0123456789ABCDEF")
    color_map = {
        i: "#" + "".join([_rng.choice(_RGB_values) for j in range(6)])
        for i in boolean_trajectory_df.index.unique().to_list()
    }

    # Create a metadata frame
    cell_colours = (
        synthetic_single_cell_experiment.kind.apply(lambda x: color_map[x])
        .to_frame()
        .rename(columns={"kind": "label_color"})
    )
    metadata = pd.concat(
        [synthetic_single_cell_experiment.kind, cell_colours], axis="columns"
    )
    metadata = metadata.rename(columns={"kind": "label"})
    # drop the number of activated genes from the synthetic expression frame
    synthetic_single_cell_experiment = synthetic_single_cell_experiment[
        synthetic_single_cell_experiment.columns[1:]
    ]

    return synthetic_single_cell_experiment, metadata
