# pylint: disable=[E0402, C)103]

from pathlib import Path
from typing import List, Dict, Tuple

from .read_write_model import Camera


def list_images(data: str) -> List[str]:
    """Lists all supported images in a directory
    Modified from:
    https://github.com/hturki/nerfstudio/nerfstudio/process_data/process_data_utils.py#L60

    Args:
        data: Path to the directory of images.
    Returns:
        Paths to images contained in the directory
    """
    data = Path(data)
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    image_paths = sorted([p for p in data.glob("[!.]*") if p.suffix.lower() in allowed_exts])
    return image_paths


def list_metadata(data: str) -> List[str]:
    """Lists all supported images in a directory
    Modified from:
    https://github.com/hturki/nerfstudio/nerfstudio/process_data/process_data_utils.py#L60

    Args:
        data: Path to the directory of images.
    Returns:
        Paths to images contained in the directory
    """
    data = Path(data)
    allowed_exts = [".pt"]
    metadata_paths = sorted([p for p in data.glob("[!.]*") if p.suffix.lower() in allowed_exts])
    return metadata_paths


def list_jsons(data: str) -> List[str]:
    """Lists all supported images in a directory
    Modified from:
    https://github.com/hturki/nerfstudio/nerfstudio/process_data/process_data_utils.py#L60

    Args:
        data: Path to the directory of images.
    Returns:
        Paths to images contained in the directory
    """
    data = Path(data)
    allowed_exts = [".json"]
    metadata_paths = sorted([p for p in data.glob("[!.]*") if p.suffix.lower() in allowed_exts])
    return metadata_paths


def read_meganerf_mappings(mappings_path: str) -> Tuple[Dict, Dict]:
    image_name_to_metadata, metadata_to_image_name = {}, {}
    with open(mappings_path, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            image_name, pt_name = line.split(',')
            pt_name = pt_name.strip()
            image_name_to_metadata[image_name] = pt_name
            metadata_to_image_name[pt_name] = image_name
            line = file.readline()

    return image_name_to_metadata, metadata_to_image_name


def get_filename_from_path(path: str) -> str:
    last_slash_index = path.rfind('/')
    return path[last_slash_index+1:]


def is_same_camera(camera1: Camera, camera2: Camera) -> bool:
    if camera1.width != camera2.width:
        return False

    if camera1.height != camera2.height:
        return False

    if len(camera1.params) != len(camera2.params):
        return False

    for i in range(len(camera1.params)):
        if camera1.params[i] != camera2.params[i]:
            return False

    return True


def get_camera_id(cameras: Dict, query_camera: Camera) -> int:
    for idx, camera in cameras.items():
        if is_same_camera(camera, query_camera):
            return idx

    return len(cameras) + 1
