from os.path import join, exists, dirname
from sklearn.neighbors import KDTree
from tool import DataProcessing as DP
from helper_ply import read_ply, write_ply
import numpy as np
import os, pickle, argparse


#     leaf_size: voxel尺寸

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/media/hello/76FCB52FFCB4EB0F/data', help='the number of GPUs to use [default: 0]')
    FLAGS = parser.parse_args()
    dataset_name = 'Muti'
    dataset_path = FLAGS.dataset_path
    dataset_path = join(dataset_path, dataset_name)
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.5
    random_sample_ratio = 10
    train_files = np.sort([join(dataset_path, 'train', i) for i in os.listdir(join(dataset_path, 'train'))])
    test_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])
    files = np.sort(np.hstack((train_files, test_files)))
    print(files)

    original_pc_folder = join(dataset_path, 'original_ply')
    os.makedirs(original_pc_folder) if not exists(original_pc_folder) else None


    for sample_type in preparation_types:
        for pc_path in files:
            cloud_typy = pc_path.split('/')[-1][-4:]
            cloud_name = pc_path.split('/')[-1][:-4]
            if cloud_typy == '.txt':
                print('start to process:', cloud_name)

                # create output directory
                # out_folder = join(dirname(dataset_path), sample_type + '_{:.3f}'.format(grid_size))
                out_folder = join(dataset_path, sample_type + '_{:.3f}'.format(grid_size))

                os.makedirs(out_folder) if not exists(out_folder) else None

                # check if it has already calculated
                # if exists(join(out_folder, cloud_name + '_KDTree.pkl')):
                #     print(cloud_name, 'already exists, skipped')
                #     continue

                if pc_path in files:
                    # xyz, labels = DP.read_ply_data(pc_path, with_rgb=True, with_rgb=False)
                    # reference = read_ply(pc_path)['reflectance']
                    # data = read_ply(pc_path)
                    data = np.loadtxt(pc_path)
                    # labels = data['scalar_Label'].astype(np.uint8)
                    # xyz = np.vstack((data['x'], data['y'], data['z'])).T
                    xyz = data[:, :3].astype((np.float32))
                    # xyz_min = np.amin(xyz, axis=0)  # 计算x y z 三个维度的最值
                    # print(xyz_min)
                    # xyz = xyz - xyz_min       # Lex local coor
                    # xyz = xyz.astype((np.float32))
                    # print(xyz[0])

                    # rgb = np.vstack((data['red'], data['green'], data['blue'])).astype(np.uint8).T
                    rgb = data[:, 3:6].astype(np.uint8)

                    # Intensity = np.expand_dims(data['scalar_Intensity'], 1).astype(np.float32)
                    # feature = np.hstack((rgb, Intensity))
                    # labels = data['scalar_Label'].astype(np.uint8)
                    labels = data[:, -1].astype(np.uint8)
                    original_pc_file = join(original_pc_folder, cloud_name + '.ply')
                    write_ply(original_pc_file, [xyz, rgb, labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                    # print(out_folder)

                    sub_ply_file = join(out_folder, cloud_name + '.ply')
                    # print(sub_ply_file)
                    if sample_type == 'grid':

                        # sub_xyz, sub_rgb, sub_labels = voxel_filter(xyz, rgb, labels, grid_size)
                        sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(xyz, rgb, labels, grid_size)
                    else:
                        sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(xyz, rgb, labels, random_sample_ratio)

                    sub_rgb = sub_rgb / 255.0
                    # sub_xyz_local = (sub_xyz - xyz_min).astype((np.float32))
                    sub_labels = np.squeeze(sub_labels)
                    write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                    search_tree = KDTree(sub_xyz, leaf_size=50)
                    kd_tree_file = join(out_folder, cloud_name + '_KDTree.pkl')
                    with open(kd_tree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                    proj_idx = proj_idx.astype(np.int32)
                    proj_save = join(out_folder, cloud_name + '_proj.pkl')
                    with open(proj_save, 'wb') as f:
                        pickle.dump([proj_idx, labels], f)
