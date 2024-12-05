#!/usr/bin/env bash


DATASET_PATH=$1
OUTPUT_PATH=$2
VOC_TREE_PATH=$3
MOST_SIMILAR_IMAGES_NUM=$4
CUDA_IDS=$5

NUM_THREADS=24
# export PYTHONDONTWRITEBYTECODE=1
# export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

COLMAP_DIR=/usr/local/bin
COLMAP_EXE=$COLMAP_DIR/colmap

mkdir $OUTPUT_PATH/sparse

$COLMAP_EXE feature_extractor \
    --database_path=$OUTPUT_PATH/database.db \
    --image_path=$DATASET_PATH/images \
    --SiftExtraction.num_threads=$NUM_THREADS \
    --SiftExtraction.use_gpu=1 \
    --SiftExtraction.gpu_index=$CUDA_IDS \
    --SiftExtraction.estimate_affine_shape=true \
    --SiftExtraction.domain_size_pooling=true \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 8192 \
    > $DATASET_PATH/log_extract_feature.txt 2>&1

$COLMAP_EXE vocab_tree_matcher \
    --database_path=$OUTPUT_PATH/database.db \
    --SiftMatching.num_threads=$NUM_THREADS \
    --SiftMatching.use_gpu=1 \
    --SiftMatching.gpu_index=$CUDA_IDS \
    --SiftMatching.guided_matching=false \
    --VocabTreeMatching.num_images=$MOST_SIMILAR_IMAGES_NUM \
    --VocabTreeMatching.num_nearest_neighbors=5 \
    --VocabTreeMatching.vocab_tree_path=$VOC_TREE_PATH \
    > $DATASET_PATH/log_match.txt 2>&1

$COLMAP_EXE mapper $OUTPUT_PATH \
    --database_path=$OUTPUT_PATH/database.db \
    --image_path=$DATASET_PATH/images \
    --output_path=$OUTPUT_PATH/sparse \
    --Mapper.num_threads=$NUM_THREADS \
    > $DATASET_PATH/log_sfm.txt 2>&1
