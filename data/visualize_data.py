import random
import json
from pathlib import Path

import numpy as np
import torch
import open3d as o3d


def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size


def visualize_one_scan(save_root, scene_name):
    inst2label_path = save_root / 'instance_id_to_label'
    pcd_path = save_root / 'pcd_with_global_alignment'
    inst_to_label = json.load(open(inst2label_path / f"{scene_name}.json"))
    pcd_data = torch.load(pcd_path / f'{scene_name}.pth')
    visualize_pcd(pcd_data, inst_to_label, vis_obj=True)


def visualize_pcd(pcd_data, inst_to_label, vis_obj=True):
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    pcds = np.concatenate([points, colors], 1)

    if not vis_obj:
        o3d_pcd = o3d.geometry.PointCloud()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.visualization.draw_geometries([mesh_frame, o3d_pcd])
        return

    obj_pcds = []
    for i in inst_to_label.keys():
        mask = instance_labels == int(i)     # time consuming
        if np.sum(mask) == 0:
            continue
        obj_pcds.append((pcds[mask], inst_to_label[i]))

    # visualize scene
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    # visualize gt
    for obj, obj_label in obj_pcds:
        gt_center, gt_size = convert_pc_to_box(obj)
        gt_o3d_box = o3d.geometry.OrientedBoundingBox(gt_center, np.eye(3,3), gt_size)
        gt_o3d_box.color = [0, 1, 0]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
        o3d.visualization.draw_geometries([o3d_pcd, gt_o3d_box, mesh_frame], window_name=obj_label)


if __name__ == "__main__":
    for dataset in ['ARKitScenes', 'HM3D', 'MultiScan', 'ProcThor', 'Structured3D', 'ScanNet', '3RScan']:
        print(dataset)
        # TODO: change dir to your own
        data_root = Path(f'/mnt/fillipo/Datasets/SceneVerse/{dataset}')
        all_scans = (data_root / 'scan_data' / 'pcd_with_global_alignment').glob('*.pth')
        scene_id = Path(random.choice(list(all_scans))).stem
        # scene_id = Path(list(all_scans)[0]).stem
        visualize_one_scan(data_root / 'scan_data', scene_id)
