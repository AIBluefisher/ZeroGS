import io
import sys
import argparse
import contextlib

from typing import Optional, List, Dict, Any
from pathlib import Path

import pycolmap


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO()) # pylint: disable=W0201
            self.out = self.capture.__enter__() # pylint: disable=W0201

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                print('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()


def run_triangulation(
    output_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    output_path.mkdir(parents=True, exist_ok=True)
    print('Running 3D triangulation...')
    if options is None:
        options = {}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, image_dir, output_path)
    return reconstruction


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:

    assert reference_model.exists(), reference_model

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / 'database.db'
    reference_model = pycolmap.Reconstruction(reference_model)

    reconstruction = run_triangulation(output_dir, database_path, image_dir, reference_model,
                                       verbose, mapper_options)
    print('Finished the triangulation with statistics:\n%s',
                reconstruction.summary())
    return reconstruction


def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find('=')
        if idx == -1:
            raise ValueError('Options format: key1=value1 key2=value2 etc.')
        key, value = arg[:idx], arg[idx+1:]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f' for {default_options.summary()}')
        value = eval(value) # pylint: disable=W0123
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(f'Incorrect type for option "{key}":'
                             f' {type(value)} vs {target_type}')
        options[key] = value
    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--reference_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args().__dict__

    # mapper_options = parse_option_args(
    #     args.pop("mapper_options"), pycolmap.IncrementalMapperOptions())
    mapper_options = pycolmap.IncrementalMapperOptions()

    main(**args, mapper_options=mapper_options)
