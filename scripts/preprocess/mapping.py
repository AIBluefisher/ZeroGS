import os
import shutil
import tqdm
import argparse

from conerf.datasets.hypersim import _collect_camera_names, _get_all_image_names


SFM_SCRIPT_PATH = os.path.join(os.getcwd(), 'scripts/preprocess/colmap_mapping.sh')
VOC_TREE_PATH = '/home/chenyu/HD_Datasets/datasets/vocab_tree_flickr100K_words256K.bin'
TOPK_IMAGES = 100
GPU_IDS = 1

# DATASETS = ['Hypersim'] #, 'nerf_synthetic'] # ['nerf_llff_data', 'ibrnet_collected_more', 'BlendedMVS']
DATASETS = ['DTU'] #, 'nerf_synthetic'] # ['nerf_llff_data', 'ibrnet_collected_more', 'BlendedMVS']
ROOT_DIR = '/home/chenyu/HD_Datasets/datasets'
# DATASETS = ['BlendedMVS']
# ROOT_DIR = '/media/chenyu/SSD_Data/datasets'


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess",
                        action="store_true",
                        help="whether to preprocess data")
    parser.add_argument("--run_colmap",
                        action="store_true",
                        help="whether to preprocess data")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)

    return parser.parse_args()


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def preprocess_nerf_synthetic_dataset(dataset_dir):
    # The DTU dataset follows pixel-nerf: https://github.com/sxyu/pixel-nerf ,
    # Url: https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR
    scenes = sorted(os.listdir(dataset_dir))
    for scene in scenes:
        scene_dir = os.path.join(dataset_dir, scene)
        image_dir = os.path.join(scene_dir, 'train')
        new_image_dir = os.path.join(scene_dir, 'images')
        
        os.system(f'cp -r {image_dir} {new_image_dir}')


def preprocess_dtu_dataset(dataset_dir):
    # The DTU dataset follows pixel-nerf: https://github.com/sxyu/pixel-nerf ,
    # Url: https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR
    scenes = sorted(os.listdir(dataset_dir))
    for scene in scenes:
        scene_dir = os.path.join(dataset_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        image_dir = os.path.join(scene_dir, 'image')
        new_image_dir = os.path.join(scene_dir, 'images')
        
        os.system(f'mv {image_dir} {new_image_dir}')


def preprocess_blended_mvs_dataset(dataset_dir):
    scenes = sorted(os.listdir(dataset_dir))
    if args.start_index < args.end_index:
        scenes = scenes[args.start_index:args.end_index]

    for scene in scenes:
        scene_dir = os.path.join(dataset_dir, scene)
        
        blended_image_dir = os.path.join(scene_dir, 'blended_images')
        image_dir = os.path.join(scene_dir, 'images')
        ori_image_dir = os.path.join(scene_dir, 'ori_images')
        masked_image_dir = os.path.join(scene_dir, 'masked_images')

        # os.system(f'rm -r {scene_dir}/output')
        # os.system(f'rm {scene_dir}/database.db {scene_dir}/poses_bounds.npy {scene_dir}/track.txt {scene_dir}/*.g2o {scene_dir}/*.json')
        # os.system(f'mv {image_dir} {ori_image_dir}')
        # os.system(f'mv {masked_image_dir} {image_dir}')

        # if not os.path.exists(image_dir):
        #     os.mkdir(image_dir)
        
        # if not os.path.exists(masked_image_dir):
        #     os.mkdir(masked_image_dir)

        # for root, dirs, files in os.walk(blended_image_dir):
        #     for file in files:
        #         image_path = os.path.join(blended_image_dir, root, file)
        #         if file.find('masked') >= 0:
        #             shutil.move(image_path, os.path.join(masked_image_dir, file))
        #         else:
        #             shutil.move(image_path, os.path.join(image_dir, file))
        
        # os.system(f'rm -r {blended_image_dir}')


def preprocess_hypersim_dataset(dataset_dir):
    scenes = sorted(os.listdir(dataset_dir))
    if args.start_index < args.end_index:
        scenes = scenes[args.start_index:args.end_index]

    pbar = tqdm.trange(len(scenes), desc="Preprocessing", leave=False)
    for scene in scenes:
        scene_dir = os.path.join(dataset_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        
        camera_names = _collect_camera_names(os.path.join(scene_dir, '_detail'))

        new_image_dir = os.path.join(scene_dir, 'images')
        origin_image_dir = os.path.join(scene_dir, 'ori_images')

        if not os.path.exists(origin_image_dir):
            os.mkdir(origin_image_dir)

        # backup
        os.system(f'mv {new_image_dir}/* {origin_image_dir}')
        # os.system(f'rm -r {origin_image_dir}/images')
        
        for i, camera_name in enumerate(camera_names):
            image_dir = os.path.join(origin_image_dir, 'scene_' + camera_name + '_final_preview')
            image_files, _ = _get_all_image_names(image_dir, image_type='tonemap')

            for image_file in image_files:
                image_name = get_filename_from_abs_path(image_file)

                sub_image_dir = os.path.join(new_image_dir, str(i))
                os.makedirs(sub_image_dir, exist_ok=True)
                new_image_file = os.path.join(sub_image_dir, image_name)
                shutil.copy(image_file, new_image_file)

        pbar.update(1)


def run_sfm(root_dir, dataset_list, args):
    for dataset in dataset_list:
        dataset_dir = os.path.join(root_dir, dataset)
        
        if args.preprocess:
            if dataset == 'BlendedMVS':
                preprocess_blended_mvs_dataset(dataset_dir)
    
            if dataset == 'DTU':
                preprocess_dtu_dataset(dataset_dir)
    
            if dataset == 'Hypersim':
                preprocess_hypersim_dataset(dataset_dir)
        
        scenes = sorted(os.listdir(dataset_dir))
        if args.start_index < args.end_index:
            scenes = scenes[args.start_index:args.end_index]

        pbar = tqdm.trange(len(scenes), desc="Running SfM", leave=False)
        for scene in scenes:
            data_dir = os.path.join(dataset_dir, scene)
            if not os.path.isdir(data_dir):
                continue
            output_dir = os.path.join(data_dir, 'sparse')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not args.run_colmap:
                # Compute bounding box.
                os.system(f'python -m scripts.preprocess.compute_bbox --colmap_dir {output_dir}/0')
                continue
            
            # print(f'output dir: {output_dir}')
            os.system(f'{SFM_SCRIPT_PATH} {data_dir} {output_dir} {VOC_TREE_PATH} {TOPK_IMAGES} {GPU_IDS}')

            shutil.move(os.path.join(output_dir, 'database.db'), os.path.join(data_dir, 'database.db'))

            # Compute bounding box.
            os.system(f'python -m scripts.preprocess.compute_bbox --colmap_dir {output_dir}/0')

            pbar.update(1)


if __name__ == '__main__':
    args = config_parser()

    run_sfm(ROOT_DIR, DATASETS, args)
