import numpy as np

import cv2
from tqdm import tqdm

from hloc.utils.database import COLMAPDatabase, blob_to_array
from hloc import logger
from hloc.utils.io import get_matches


def import_matches(image_ids, database_path, pairs_path, matches_path,
                   min_match_score=None, skip_geometric_verification=False
) -> int :
    logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()
    return len(pairs)


def read_camera_intrinsics_by_image_id(image_id: int, db: COLMAPDatabase):
    rows = db.execute(f'SELECT camera_id FROM images WHERE image_id={image_id}')
    camera_id = next(rows)[0]
    rows = db.execute(f'SELECT params FROM cameras WHERE camera_id={camera_id}')
    params = blob_to_array(next(rows)[0], dtype=np.float64)

    # FIXME(chenyu): when camera model is not a simple pinhole.
    intrinsics = np.zeros((3, 3), dtype=np.float64)
    intrinsics[0, 0] = intrinsics[1, 1] = params[0]
    intrinsics[0, 2], intrinsics[1, 2] = params[1], params[2]
    intrinsics[2, 2] = 1.
    return intrinsics


def read_all_keypoints(db: COLMAPDatabase):
    keypoints_dict = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))
    return keypoints_dict


def extract_inlier_keypoints_pair(inlier_matches, keypoints1, keypoints2):
    inlier_keypoints1, inlier_keypoints2 = [], []
    num_inliers = inlier_matches.shape[0]
    for i in range(num_inliers):
        idx = inlier_matches[i]
        inlier_keypoints1.append(keypoints1[idx[0]])
        inlier_keypoints2.append(keypoints2[idx[1]])
    
    inlier_keypoints1 = np.stack(inlier_keypoints1, axis=0)
    inlier_keypoints2 = np.stack(inlier_keypoints2, axis=0)
    return inlier_keypoints1, inlier_keypoints2


def triangulate(inlier_keypoints1, inlier_keypoints2,
                extrinsics1: np.ndarray, extrinsics2: np.ndarray,
                intrinsics1: np.ndarray, intrinsics2: np.ndarray):
    proj_mtx1 = np.matmul(intrinsics1, extrinsics1)
    proj_mtx2 = np.matmul(intrinsics2, extrinsics2)

    points3d = cv2.triangulatePoints(projMatr1=proj_mtx1, projMatr2=proj_mtx2,
                                     projPoints1=inlier_keypoints1.transpose(1, 0),
                                     projPoints2=inlier_keypoints2.transpose(1, 0))
    points3d = points3d.transpose(1, 0)
    points3d = points3d[:, :3] / points3d[:, 3].reshape(-1, 1)
    return points3d


def compute_depth(proj_matrix, point3d):
    homo_point3d = np.ones(4)
    homo_point3d[0:3] = point3d
    proj_z = np.dot(proj_matrix[2, :].T, homo_point3d)
    return proj_z * np.linalg.norm(proj_matrix[:, 2], ord=2)


def check_cheirality(inlier_keypoints1, inlier_keypoints2,
                     extrinsic1: np.ndarray, extrinsic2: np.ndarray,
                     intrinsics1: np.ndarray, intrinsics2: np.ndarray):
    min_depth = 1e-16
    max_depth = 1000 * np.linalg.norm(
        np.dot(extrinsic2[:3, :3].T, extrinsic2[:, 3]), ord=2)
    points3d = []

    tmp_points3d = triangulate(inlier_keypoints1, inlier_keypoints2,
                               extrinsic1, extrinsic2, intrinsics1, intrinsics2)
    for point3d in tmp_points3d:
        # Checking for positive depth in front of both cameras.
        depth1 = compute_depth(extrinsic1, point3d)
        if depth1 < max_depth and depth1 > min_depth:
          depth2 = compute_depth(extrinsic2, point3d)
          if depth2 < max_depth and depth2 > min_depth:
            points3d.append(point3d)
      
    return points3d


def decompose_essential_matrix(
    keypoints1, keypoints2,
    essential_matrix, inlier_matches,
    intrinsics1: np.ndarray, intrinsics2: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Assume that the image_id1 is at [I|0] and second image_id2 is at [R|t] 
    where R, t are derived from the essential matrix.

    Args:
        keypoints1: keypoints locations of image1
        keypoints1: keypoints locations of image2
        essential_matrix: 3 x 3 numpy array,
        inlier_matches: matched keypoints indices between image1 and image2
        intrinsics1: 3 x 3 numpy array for image1
        intrinsics2: 3 x 3 numpy array for image2

    Returns:
        extrinsic matrix of shape (1, 12) from image 1 to image 2
    """

    inlier_keypoints1, inlier_keypoints2 = extract_inlier_keypoints_pair(
        inlier_matches, keypoints1, keypoints2)
    # print(f'{inlier_keypoints1.shape}')
    # print(f'{inlier_keypoints2.shape}')

    extrinsic1 = np.zeros(shape=[3, 4], dtype=np.float64)
    extrinsic1[:3, :3] = np.eye(3)
    # relative motion from camera1 to camera2.
    extrinsics2 = np.zeros(shape=[3, 4], dtype=np.float64)

    W = np.zeros((3, 3))
    W[0, 1], W[1, 0], W[2, 2] = -1, 1, 1
    U, _, Vh = np.linalg.svd(essential_matrix)

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vh) < 0:
        Vh *= -1

    R1, R2 = np.dot(np.dot(U, W), Vh), np.dot(np.dot(U, np.transpose(W)), Vh)
    t = U[:, 2]
    t /= np.linalg.norm(t, ord=2)

    def compose_projection_matrix(R, t):
        P = np.zeros(shape=[3, 4], dtype=float)
        P[:3, :3], P[:, 3] = R, t
        return P
    
    # Generate candidate projection matrices.
    P2_list = []
    P2_list.append(compose_projection_matrix(R1,  t))
    P2_list.append(compose_projection_matrix(R2,  t))
    P2_list.append(compose_projection_matrix(R1, -t))
    P2_list.append(compose_projection_matrix(R2, -t))
    
    candidate_points3d, points3d = [], []
    # Then, we need to iterate over each projection matrix and 
    # make the cheirality validation.
    for extrinsic2 in P2_list:
        candidate_points3d = check_cheirality(
            inlier_keypoints1, inlier_keypoints2,
            extrinsic1, extrinsic2, intrinsics1, intrinsics2)
        # print(f'len points3d: {len(points3d)}')
        if len(points3d) < len(candidate_points3d):
            points3d[:] = candidate_points3d
            extrinsics2[:] = extrinsic2

    # print(f'final len points3d: {len(points3d)}')
    if len(points3d) == 0:
        return None, None
    
    points3d = np.stack(points3d, axis=0)

    return extrinsics2.reshape(1, -1), points3d
