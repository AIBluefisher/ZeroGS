#!/usr/bin/env bash

# Run triangulator with known camera poses.

COLMAP_DIR=/usr/local/bin
COLMAP_EXE=$COLMAP_DIR/colmap

export PYTHONDONTWRITEBYTECODE=1

PROJECT_PATH=$1
colmap_method=$2 # ['colmap', 'pycolmap']

if [ `echo $colmap_method | grep -c "py" ` -gt 0 ]
then
    HOME_DIR=$HOME
    CODE_ROOT_DIR=$HOME/'Projects/ZeroGS'
    cd $CODE_ROOT_DIR

    python -m scripts.preprocess.hloc_mapping.triangulate_from_existing_model \
        --sfm_dir $PROJECT_PATH \
        --reference_model $PROJECT_PATH/sparse/triangulator_input \
        --output_dir $PROJECT_PATH/sparse/0 \
        --image_dir $PROJECT_PATH \
        --verbose \
        > $PROJECT_PATH/log_triangulate.txt 2>&1
else
    $COLMAP_EXE point_triangulator \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH \
        --input_path $PROJECT_PATH/sparse/triangulator_input \
        --output_path $PROJECT_PATH/sparse/0 \
        > $PROJECT_PATH/log_triangulate.txt 2>&1
fi
