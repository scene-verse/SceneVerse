g


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


def load_scan(pcd_path, inst2label_path, scene_name):
    pcd_data = torch.load(pcd_path / f'{scene_name}.pth')
    inst_to_label = torch.load(inst2label_path / f"{scene_name}.pth")
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    pcds = np.concatenate([points, colors], 1)
    return points, colors, pcds, instance_labels, inst_to_label


def visualize_one_scene(obj_pcds, points, colors, caption):
    # visualize scene
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points)
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    # visualize gt
    for idx, (obj, obj_label) in enumerate(obj_pcds):
        if idx > 3:
            break
        gt_center, gt_size = convert_pc_to_box(obj)
        gt_o3d_box = o3d.geometry.OrientedBoundingBox(gt_center, np.eye(3,3), gt_size)
        gt_o3d_box.color = [0, 1, 0]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
        o3d.visualization.draw_geometries([o3d_pcd, gt_o3d_box, mesh_frame], window_name=obj_label+'_'+caption)


def visualize_data(save_root, scene_name, vis_obj=True):
    inst2label_path = save_root / 'instance_id_to_label'
    pcd_path = save_root / 'pcd_with_global_alignment'

    points, colors, pcds, instance_labels, inst_to_label = load_scan(pcd_path, inst2label_path, scene_name)

    if not vis_obj:
        o3d_pcd = o3d.geometry.PointCloud()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.visualization.draw_geometries([mesh_frame, o3d_pcd])
        return

    obj_pcds = []
    for i in inst_to_label.keys():
        mask = instance_labels == i     # time consuming
        if np.sum(mask) == 0:
            continue
        obj_pcds.append((pcds[mask], inst_to_label[i]))

    visualize_one_scene(obj_pcds, points, colors, scene_name)

def visualize_refer(save_root, anno_file):
    inst2label_path = save_root / 'instance_id_to_label'
    pcd_path = save_root / 'pcd_with_global_alignment'
    json_data = json.load(open(anno_file, 'r', encoding='utf8'))
    for item in json_data:
        scan_id = item['scan_id']
        inst2label_path = save_root / 'instance_id_to_label'
        pcd_path = save_root / 'pcd_with_global_alignment'

        inst_to_label = torch.load(inst2label_path / f"{scan_id}.pth")
        pcd_data = torch.load(pcd_path / f'{scan_id}.pth')
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        pcds = np.concatenate([points, colors], 1)

        target_id = int(item['target_id'])
        mask = instance_labels == target_id
        if np.sum(mask) == 0:
            continue

        obj_pcds = [(pcds[mask], inst_to_label[target_id])]
        visualize_one_scene(obj_pcds, points, colors, scan_id+'-'+item['utterance'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", required=True, type=str, help="path of dataset dir")
    parser.add_argument("-d", "--dataset", type=str,
                        help="available datasets in ['ARKitScenes', 'HM3D', 'MultiScan', 'ProcThor', \
                        'Structured3D', 'ScanNet', '3RScan']")
    parser.add_argument("--vis_refer", action="store_true",
                        help="visualize reference data")
    parser.add_argument("-a", "--anno", type=str, default="ssg_ref_rel2_template.json",
                        help="the annotation file for reference")
    args = parser.parse_args()
    dataset = args.dataset
    assert dataset in ['ARKitScenes', 'HM3D', 'MultiScan', 'ProcThor', 'Structured3D', 'ScanNet', '3RScan']
    print(dataset)
    data_root = Path(args.root) / dataset
    if args.vis_refer:
        if dataset == 'ScanNet':
            anno_file = data_root / 'annotations/refer' / args.anno
        else:
            anno_file = data_root / 'annotations' / args.anno

        visualize_refer(data_root / 'scan_data', anno_file)
    else:
        all_scans = (data_root / 'scan_data' / 'pcd_with_global_alignment').glob('*.pth')
        scene_id = Path(random.choice(list(all_scans))).stem
        visualize_data(data_root / 'scan_data', scene_id)