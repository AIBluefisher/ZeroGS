import sys
import sqlite3

from typing import Dict

import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

#-------------------------------------------------------------------------------
# create table commands

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""

CREATE_INLIER_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([CREATE_CAMERAS_TABLE, CREATE_DESCRIPTORS_TABLE,
    CREATE_IMAGES_TABLE, CREATE_INLIER_MATCHES_TABLE, CREATE_KEYPOINTS_TABLE,
    CREATE_MATCHES_TABLE, CREATE_NAME_INDEX])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()

    return np.getbuffer(array)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initialize_tables = lambda: self.executescript(CREATE_ALL)

        self.initialize_cameras = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.initialize_descriptors = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.initialize_images = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.initialize_inlier_matches = \
            lambda: self.executescript(CREATE_INLIER_MATCHES_TABLE)
        self.initialize_keypoints = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.initialize_matches = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)

        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid


def fetch_images_from_database(database_path: str) -> Dict:
    db = COLMAPDatabase.connect(database_path) # pylint: disable=[C0103]
    rows = db.execute("SELECT * FROM images")
    name_to_image_id = {}
    for row in rows:
        image_id, name = row[0], row[1]
        # print(f'image_id: {image_id}, name: {name}')
        name_to_image_id[name] = image_id

    return name_to_image_id
