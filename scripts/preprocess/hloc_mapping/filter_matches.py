from pathlib import Path
import argparse
import numpy as np
import networkx as nx

from matplotlib import pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

from disambiguation.utils.read_write_database import remove_matches_from_db
from disambiguation.utils.run_colmap import run_matches_importer


def draw_graph(scores, plot_path, display=False):
    graph = nx.from_numpy_array(scores)
    # print(scores)
    pos = nx.nx_agraph.graphviz_layout(graph)
    edge_vmin = np.percentile(scores[scores.nonzero()], 10)
    edge_vmax = np.percentile(scores[scores.nonzero()], 90)
    # print(edge_vmin, edge_vmax)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        edge_color=[graph[u][v]['weight'] for u, v in graph.edges],
        # edge_cmap=plt.cm.plasma,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax)
    plt.savefig(plot_path)
    if display:
        plt.show()
    plt.close()
    return


def filter_with_fixed_threshold(scores, thres, plot_path=None):
    valid = scores >= thres
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=False)
    return valid


def filter_with_knn(scores, k, plot_path):
    valid = np.zeros_like(scores, dtype=np.bool)
    valid_indices = scores.argsort()[:, -k:]
    for i in range(scores.shape[0]):
        for j in valid_indices[i]:
            valid[i, j] = True
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=False)
    return valid


def filter_with_mst_min(scores, plot_path=None):
    min_scores = np.minimum(scores, scores.T)
    assert np.allclose(min_scores, min_scores.T)
    mst = minimum_spanning_tree(-min_scores)
    valid = (-mst).toarray() > 0
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=False)
    return valid


def filter_with_mst_mean(scores, plot_path=None):
    mean_scores = (scores + scores.T) / 2
    assert np.allclose(mean_scores, mean_scores.T)
    mst = minimum_spanning_tree(-mean_scores)
    valid = (-mst).toarray() > 0
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=False)
    return valid


def filter_with_percentile(scores, percentile, plot_path=None):
    num_images = scores.shape[0]
    thres = np.zeros((num_images, 1))
    for i in range(num_images):
        thres[i] = np.percentile(scores[i, scores[i].nonzero()], percentile)
    valid = scores >= thres
    invalid = np.logical_not(valid)
    scores[invalid] = 0.
    if plot_path is not None:
        draw_graph(scores, plot_path, display=False)
    return valid


def main(colmap_path: str,
         results_path: str,
         filter_type: str,
         threshold: float,
         scores_dir: Path,
         scores_name: str,
         topk: int,
         percentile: float,
         old_db_path: str,
         new_db_path: str,
         geometric_verification_type: str):
    scores_path = scores_dir / scores_name
    scores = np.load(scores_path)

    # valid = scores >= args.threshold
    if filter_type == 'threshold':
        assert threshold is not None
        output_path = results_path / ('sparse' + scores_name[6:-4] +
                                      f'_t{threshold:.2f}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        match_list_path = results_path / (
            'match_list' + scores_name[6:-4] + f'_t{threshold}.txt')
        valid = filter_with_fixed_threshold(scores, threshold, plot_path)
    elif filter_type == 'knn':
        assert topk is not None
        output_path = results_path / ('sparse' + scores_name[6:-4] +
                                      f'_k{topk}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        match_list_path = results_path / (
            'match_list' + scores_name[6:-4] + f'_k{topk}.txt')
        valid = filter_with_knn(scores, topk, plot_path)
    elif filter_type == 'percentile':
        assert percentile is not None
        output_path = results_path / ('sparse' + scores_name[6:-4] +
                                      f'_p{percentile}')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        match_list_path = results_path / (
            'match_list' + scores_name[6:-4] + f'_p{percentile}.txt')
        valid = filter_with_percentile(scores, percentile, plot_path)
    elif filter_type == 'mst_min':
        output_path = results_path / ('sparse' + scores_name[6:-4] +
                                      '_mst_min')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        match_list_path = results_path / (
            'match_list' + scores_name[6:-4] + '_mst_min.txt')
        valid = filter_with_mst_min(scores, plot_path)
        # we don't do reconstruction based with mst graph as it is too sparse.
        # use it for visualization only
        exit(0)
    elif filter_type == 'mst_mean':
        output_path = results_path / ('sparse' + scores_name[6:-4] +
                                      '_mst_mean')
        output_path.mkdir(exist_ok=True)
        plot_path = output_path / 'match_graph.png'
        match_list_path = results_path / (
            'match_list' + scores_name[6:-4] + '_mst_mean.txt')
        valid = filter_with_mst_mean(scores, plot_path)
        # we don't do reconstruction based with mst graph as it is too sparse.
        # use it for visualization only
        exit(0)
    else:
        raise NotImplementedError

    remove_matches_from_db(old_db_path, new_db_path, match_list_path, valid)
    run_matches_importer(colmap_path,
                         new_db_path,
                         match_list_path,
                         use_gpu=False,
                         colmap_matching_type=geometric_verification_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--results_path', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--scores_name', type=str, required=True,
                        default='yan', choices=['yan', 'cui'])
    parser.add_argument('--filter_type',
                        type=str,
                        choices=['threshold', 'knn', 'mst_min', 'mst_mean', 'percentile'])
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--percentile', type=float)
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--old_db_path', type=str, Required=True)
    parser.add_argument('--new_db_path', type=str, Required=True)
    parser.add_argument('--geometric_verification_type',
                        type=str,
                        required=True,
                        choices=['default', 'strict'])


    args = parser.parse_args()

    main(args.colmap_path, args.results_path, args.filter_type, args.threshold,
         args.scores_name, args.topk, args.percentile, args.old_db_path,
         args.new_db_path, args.geometric_verification_type)
