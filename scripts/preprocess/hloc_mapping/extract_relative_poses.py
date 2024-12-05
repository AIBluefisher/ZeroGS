# pylint: disable=[C0103]
import os
from pathlib import Path
from pprint import pformat
import argparse
import math

import torch
import numpy as np
from tqdm import tqdm

import pycolmap
from hloc.utils import viz_3d
from hloc.utils.database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from hloc.triangulation import import_features, estimation_and_geometric_verification
# from disambiguation import calculate_geodesic_consistency_scores

from scripts.preprocess.hloc_mapping import extract_features, match_features, \
    pairs_from_retrieval, reconstruction, filter_matches
from scripts.preprocess.hloc_mapping.utils import read_all_keypoints, import_matches, \
    decompose_essential_matrix, read_camera_intrinsics_by_image_id
# from dbarf.geometry.rotation import Rotation


class Quaternion:
    def __init__(self, quat: np.ndarray) -> None:
        self.quat = quat

    @classmethod
    def normalize(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The normalized quaternion such that
                qw * qw + qx * qx + qy * qy + qz * qz = 1.
        """
        # qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        norm = np.linalg.norm(quat)
        normalized_quat = quat[:] / norm
        return normalized_quat

    @classmethod
    def inverse(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The quaternion in the opposite direction.
        """
        quat = Quaternion.normalize(quat)
        quat[1:] = -quat[1:]
        return quat

    @classmethod
    def to_rotation_matrix(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The 3*3 rotation matrix.
        """
        quat = Quaternion.normalize(quat)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        R = np.zeros((3, 3), dtype=quat.dtype)
        R[0, 0] = qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2
        R[0, 1] = 2. * (qx * qy - qw * qz)
        R[0, 2] = 2. * (qx * qz + qw * qy)
        R[1, 0] = 2. * (qx * qy + qw * qz)
        R[1, 1] = qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2
        R[1, 2] = 2. * (qy * qz - qw * qx)
        R[2, 0] = 2. * (qx * qz - qw * qy)
        R[2, 1] = 2. * (qy * qz + qw * qx)
        R[2, 2] = qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2

        return R

    @classmethod
    def to_angle_axis(cls, quat: np.ndarray) -> np.ndarray:
        # NOTE: quaternion should be normalized!
        quat = Quaternion.normalize(quat)
        angle_axis = np.zeros(3)

        qx, qy, qz = quat[1], quat[2], quat[3]
        sin_squared_theta = qx * qx + qy * qy + qz * qz

        # For quaternions representing non-zero rotation, the conversion
        # is numerically stable.
        if sin_squared_theta > 0.0:
            sin_theta = np.sqrt(sin_squared_theta)
            cos_theta = quat[0]

            # If cos_theta is negative, theta is greater than pi/2, which means
            # that angle for the angle_axis vector which is 2*theta would be greater
            # than pi.
            #
            # While this will result in the correct rotation, it does not result in
            # a normalized angle-axis vector.
            # In that case we observe that 2*theta ~ 2*theta - 2*pi, which is
            # equivalent saying:
            #
            # theta - pi = atan(sin(theta - pi), cos(theta - pi))
            #            = atan(-sin(theta), -cos(theta))
            #
            two_theta = 2. * np.arctan2(-sin_theta, -cos_theta) \
                        if cos_theta < 0. else 2. * np.arctan2(sin_theta, cos_theta)
            k = two_theta / sin_theta
            angle_axis[0], angle_axis[1], angle_axis[2] = k * qx, k * qy, k * qz
        else:
            # For zero rotation, sqrt() will produce NaN in the derivative since
            # the argument is zero. By approximating with a Taylor series, and
            # truncating at one term, the value and first derivatives will be computed
            # correctly.
            k = 2.
            angle_axis[0], angle_axis[1], angle_axis[2] = k * qx, k * qy, k * qz

        return angle_axis

    def rotate_point(self, point3D):
        pass


class AngleAxis:
    def __init__(self, angle_axis: np.ndarray) -> None:
        self.rotation_vec = angle_axis

    # @classmethod
    # def normalize(cls, angle_axis: np.ndarray):
    #     norm = np.linalg.norm(angle_axis)
    #     normalized_angle_axis = angle_axis[:] / norm
    #     return normalized_angle_axis

    @classmethod
    def to_rotation_matrix(cls, angle_axis: np.ndarray):
        a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        theta_squared = a0 ** 2 + a1 ** 2 + a2 ** 2
        R = np.zeros((3, 3), dtype=angle_axis.dtype)
        min_threshold = 1e-15

        # We need to be careful to only evaluate the square root if the norm of the
        # rotation vector is greater than zero. Otherwise, we get a division by zero.
        if theta_squared > min_threshold:
            theta = np.sqrt(theta_squared)
            wx, wy, wz = a0 / theta, a1 / theta, a2 / theta

            sin_theta, cos_theta = np.sin(theta), np.cos(theta)

            # The equation is derived from the Rodrigues formula.
            R[0, 0] = cos_theta + wx * wx * (1. - cos_theta)
            R[1, 0] = wz * sin_theta + wx * wy * (1. - cos_theta)
            R[2, 0] = -wy * sin_theta + wx * wz * (1. - cos_theta)
            R[0, 1] = wx * wy * (1. - cos_theta) - wz * sin_theta
            R[1, 1] = cos_theta + wy * wy * (1. - cos_theta)
            R[2, 1] = wx * sin_theta + wy * wz * (1. - cos_theta)
            R[0, 2] = wy * sin_theta + wx * wz * (1. - cos_theta)
            R[1, 2] = -wx * sin_theta + wy * wz * (1. - cos_theta)
            R[2, 2] = cos_theta + wz * wz * (1. - cos_theta)
        else:
            # Near zero, we switch to using the first order Taylor expansion.
            R[0, 0], R[1, 0], R[2, 0] = 1., a2, -a1
            R[0, 1], R[1, 1], R[2, 1] = -a2, 1., a0
            R[0, 2], R[1, 2], R[2, 2] = a1, -a0, 1.

        return R

    @classmethod
    def to_quaternion(cls, angle_axis: np.ndarray):
        quat = np.zeros(4, dtype=angle_axis.dtype)

        a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        # theta = np.linalg.norm(angle_axis)
        theta_squared = a0 ** 2 + a1 ** 2 + a2 ** 2

        # For points not at the origin, the full conversion is numerically stable.
        if theta_squared > 0.:
            theta = np.sqrt(theta_squared)
            half_theta = theta / 2
            k = np.sin(half_theta) / theta
            quat[0] = np.cos(half_theta)
            quat[1], quat[2], quat[3] = k * a0, k * a1, k * a2
        else:
            # At the origin,sqrt will produce NaN in the derivative since
            # the argument is zero. By approximating with a Taylor series,
            # and truncating at one term, the value and first derivatives
            # will be computed correctly.
            k = 0.5
            quat[0], quat[1], quat[2], quat[3] = 1., k * a0, k * a1, k * a2

        return quat

    @classmethod
    def theta(cls, angle_axis: np.ndarray):
        # angle_axis = AngleAxis.normalize(angle_axis)
        # a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        return np.linalg.norm(angle_axis)

    def rotation_point(self, point3D):
        pass


class Rotation:
    def __init__(self, R: np.ndarray) -> None:
        self.R = R

    @classmethod
    def to_quaternion(cls, R):
        # Ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        quat = np.zeros(4, dtype=R.dtype)
        R00, R01, R02 = R[0, 0], R[0, 1], R[0, 2]
        R10, R11, R12 = R[1, 0], R[1, 1], R[1, 2]
        R20, R21, R22 = R[2, 0], R[2, 1], R[2, 2]
        trace = R00 + R11 + R22

        r = math.sqrt(1 + trace)
        qw = .5 * r
        qx = np.sign(R21 - R12) * .5 * math.sqrt(1. + R00 - R11 - R22)
        qy = np.sign(R02 - R20) * .5 * math.sqrt(1. - R00 + R11 - R22)
        qz = np.sign(R10 - R01) * .5 * math.sqrt(1. - R00 - R11 + R22)
        quat[0], quat[1], quat[2], quat[3] = qw, qx, qy, qz

        # If the matrix contains significant error, such as accumulated numerical
        # error, we may construct a symmetric 4*4 matrix.
        if np.isnan(quat).any():
            K = np.array([[R00 - R11 - R22, R10 + R01, R20 + R02, R21 - R12],
                          [R10 + R01, R11 - R00 - R22, R21 + R12, R02 - R20],
                          [R20 + R02, R21 + R12, R22 - R00 - R11, R10 - R01],
                          [R21 - R12, R02 - R20, R10 - R01, R00 + R11 + R22]], dtype=R.dtype)
            eigen_values, eigen_vecs = np.linalg.eigh(K)
            qx, qy, qz, qw = eigen_vecs[eigen_values.argmax()]
            quat[0], quat[1], quat[2], quat[3] = qw, qx, qy, qz
            if quat[0] < 0:
                quat *= -1

        return quat

    @classmethod
    def to_angle_axis(cls, R):
        # We do not compute the angle axis by the Rodrigues formula.
        quat = Rotation.to_quaternion(R)
        angle_axis = Quaternion.to_angle_axis(quat)

        return angle_axis

    def transpose(self):
        return self.R.t()

    def chordal_distance(self, Q: np.ndarray):
        R_diff = self.R - Q
        return np.linalg.norm(R_diff)

    def angular_distance(self, Q: np.ndarray):
        relative_rotation = self.transpose() * Q
        angle_axis = Rotation.to_angle_axis(relative_rotation)
        return AngleAxis.theta(angle_axis)

    def rotate_point(self, point3D):
        return np.dot(self.R, point3D)

    def left_multiply(self, R: np.ndarray):
        self.R = self.R @ R

    def right_multiply(self, R: np.ndarray):
        self.R = R @ self.R


def disambiguate_via_geodesic_consistency(database_path,
                                          track_degree,
                                          coverage_thres, alpha, minimal_views,
                                          ds):
    print('disambiguate via geodesic consistency: ')
    calculate_geodesic_consistency_scores.main(database_path, track_degree,
                                               coverage_thres, alpha,
                                               minimal_views, ds)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_matches', type=int, default=30,
                        help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--disambiguate', action="store_true",
                        help='Enable/Disable disambiguating wrong matches.')
    parser.add_argument('--track_degree', type=int, default=3)
    parser.add_argument('--coverage_thres', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--minimal_views', type=int, default=5)
    parser.add_argument('--ds', type=str,
                        choices=['dict', 'smallarray', 'largearray'],
                        default='largearray')
    parser.add_argument('--filter_type', type=str, choices=[
                        'threshold', 'knn', 'mst_min', 'mst_mean', 'percentile'],
                        default='threshold')
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--percentile', type=float)
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--geometric_verification_type',
                        type=str,
                        choices=['default', 'strict'],
                        default='default')
    parser.add_argument('--recon', action="store_true",
                        help='Indicates whether to reconstruct the scene.')
    parser.add_argument('--visualize', action="store_true",
                        help='Whether to visualize the reconstruction.')
    parser.add_argument('--gpu_idx', type=str, default='0')
    args = parser.parse_args()
    return args


def extract_relative_poses(
    database_path: Path, visualization=True
) -> (np.ndarray, np.ndarray):
    db = COLMAPDatabase.connect(database_path)
    keypoints = read_all_keypoints(db)
    # rows = db.execute("SELECT * FROM two_view_geometries")
    cursor = db.cursor()
    cursor.execute("SELECT * FROM two_view_geometries")
    num_match_pairs = 0
    for _ in cursor:
        num_match_pairs += 1

    relative_motions = []
    pbar = tqdm(total=num_match_pairs)

    print(f'num two view geometries: {num_match_pairs}')
    cursor.execute("SELECT * FROM two_view_geometries")
    for rows in cursor:
        pbar.set_description('Extracting relative poses ')
        pbar.update(1)

        # pair_id, shape1, shape2, matches, config, F, E, H, qvec, tvec = next(rows)
        pair_id, shape1, shape2, matches, config, F, E, H, qvec, tvec = rows # pylint: disable=W0612
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)

        if matches is None:
            continue

        matches = blob_to_array(matches, np.uint32).reshape(shape1, shape2)

        # FIXME(chenyu): in some cases, we need to decompose homography matrix rather than
        # the essential matrix. There should be some strategies to do this inside COLMAP.
        E = blob_to_array(E, np.float64).reshape(3, 3)

        intrinsics1 = read_camera_intrinsics_by_image_id(image_id1, db)
        intrinsics2 = read_camera_intrinsics_by_image_id(image_id2, db)

        relative_motion, points3d = decompose_essential_matrix(
            keypoints[image_id1], keypoints[image_id2], E, matches,
            intrinsics1, intrinsics2)

        if not (relative_motion is None or points3d is None):
            # image1, image2, [rotations, translations]
            relative_motion = np.concatenate(
                (np.array([[image_id1, image_id2]]), relative_motion), axis=1)
            relative_motions.append(relative_motion)

    if visualization is True:
        fig = viz_3d.init_figure()
        viz_3d.plot_points(fig, points3d)
        fig.show()

    return relative_motions


def main(args):
    # List the standard configurations available.
    print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    device = 'cuda:' + args.gpu_idx if torch.cuda.is_available() else 'cpu'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    # feature_conf = extract_features.confs['sift']
    # matcher_conf = match_features.confs['NN-ratio']

    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)

    # the SfM model we will build.
    sfm_dir = Path(os.path.join(args.outputs, 'sfm_superpoint+superglue'))
    # sfm_dir = Path(os.path.join(args.outputs, 'sfm_sift+nn-ratio'))
    # top-k retrieved by NetVLAD.
    match_pairs = Path(os.path.join(sfm_dir, 'pairs-netvlad.txt'))

    local_features_path = extract_features.main(
        feature_conf, args.dataset_dir, sfm_dir, device=device)

    global_descriptors_path = extract_features.main(
        retrieval_conf, args.dataset_dir, sfm_dir, device=device)
    pairs_from_retrieval.main(
        global_descriptors_path, match_pairs, args.num_matches, device=device)

    matches_path = match_features.main(
        matcher_conf, match_pairs, feature_conf['output'], sfm_dir, device=device)

    assert local_features_path.exists(), local_features_path
    assert match_pairs.exists(), match_pairs
    assert matches_path.exists(), matches_path

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / 'database.db'

    reconstruction.create_empty_db(database_path)
    reconstruction.import_images(
        args.dataset_dir, database_path, camera_mode=pycolmap.CameraMode.AUTO)
    image_ids = reconstruction.get_image_ids(database_path)
    import_features(image_ids, database_path, local_features_path)
    import_matches(
        image_ids, database_path, match_pairs, matches_path,
        min_match_score=None, skip_geometric_verification=False)
    estimation_and_geometric_verification(database_path, match_pairs, verbose=False)

    if args.disambiguate is True:
        print('Disambiguating Wrong Matches.')
        disambiguate_via_geodesic_consistency(database_path,
                                              args.track_degree,
                                              args.coverage_thres, args.alpha,
                                              args.minimal_views, args.ds)

        filtered_db_path = sfm_dir / 'disambig_database.db'
        scores_dir = database_path.parent
        scores_name = f'scores_yan_t{args.track_degree}_c' + \
                      f'{args.coverage_thres}_a{args.alpha}_m{args.minimal_views}.npy'
        filter_matches.main(
            args.colmap_path, sfm_dir, args.filter_type, args.threshold,
            scores_dir, scores_name, args.topk, args.percentile, database_path,
            filtered_db_path, args.geometric_verification_type)

        # Overwrite original database path.
        database_path = filtered_db_path

    # The relative rotations and translations are not stored in COLMAP's database.
    relative_poses = extract_relative_poses(database_path, args.visualize)
    view_graph_path = Path(args.outputs) / f'VG_N{len(image_ids)}_M{len(relative_poses)}.g2o'

    model = None
    if args.recon:
        model = reconstruction.main(
            database_path,
            sfm_dir, args.dataset_dir, match_pairs,
            local_features_path, matches_path)
        if args.visualize:
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig, model)
            fig.show()

    print(f'Local features are extracted to: {local_features_path}')
    print(f'matches are saved to: {matches_path}')
    print(f'view graph is saved to {view_graph_path}')
    return view_graph_path, database_path, len(relative_poses)


if __name__ == '__main__':
    args = parse_args()
    main(args)
