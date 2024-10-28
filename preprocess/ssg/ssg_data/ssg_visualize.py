import open3d as o3d
import numpy as np
import torch


def vis_dataset(ObjNode_dict, relation, scene_path, scan_id, scene_center):
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
    pcd_data = torch.load(scene_path / 'pcd_with_global_alignment' / f'{scan_id}.pth')
    points, colors, _ = pcd_data[0], pcd_data[1], pcd_data[-1]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    scene_show = [o3d_pcd]

    np.random.shuffle(relation)
    for rel in relation:
        if len(rel) == 3:
            if rel[1] == -2:
                src = ObjNode_dict[rel[0]]
                gt_o3d_box_src = o3d.geometry.OrientedBoundingBox(src.position + scene_center, np.eye(3, 3),
                                                                  src.size)
                gt_o3d_box_src.color = [0, 1, 0]
                obj_label = f'''{src.label} {rel[2]}'''
                bbox_show_list = [gt_o3d_box_src]
                o3d.visualization.draw_geometries(scene_show + [coordinate] + bbox_show_list,
                                                  window_name=obj_label)

            else:
                src = ObjNode_dict[rel[0]]
                tgt = ObjNode_dict[rel[1]]


                gt_o3d_box_src = o3d.geometry.OrientedBoundingBox(src.position + scene_center, np.eye(3, 3),
                                                                  src.size)
                gt_o3d_box_src.color = [0, 1, 0]
                gt_o3d_box_tgt = o3d.geometry.OrientedBoundingBox(tgt.position + scene_center, np.eye(3, 3),
                                                                  tgt.size)
                gt_o3d_box_tgt.color = [1, 0, 0]

                obj_label = f'''{tgt.label} - {rel[2]} - {src.label} '''
                bbox_show_list = [gt_o3d_box_src, gt_o3d_box_tgt]

                o3d.visualization.draw_geometries(scene_show + [coordinate] + bbox_show_list,
                                                  window_name=obj_label)
        else:
            tgts = [ObjNode_dict[tgt] for tgt in rel[0]]
            gt_o3d_box_tgts = [o3d.geometry.OrientedBoundingBox(tgt.position + scene_center, np.eye(3, 3), tgt.size)
                               for tgt in tgts]
            # gt_o3d_box_tgt.color = [1, 0, 0]

            obj_label = f''' {ObjNode_dict[tgts[0].id].label} {rel[1]}'''
            bbox_show_list = [gt_o3d_box_tgts]
            o3d.visualization.draw_geometries(scene_show + [coordinate] + bbox_show_list,
                                              window_name=obj_label)
