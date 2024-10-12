import json
import pickle
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import torch
import networkx as nx
import numpy as np

import ssg_utils as utils
from ssg_data import dictionary
from ssg_data.ssg_visualize import vis_dataset
from ssg_data.script.ObjNode import ObjNode
from relationships.support import cal_support_relations
from relationships.proximity import cal_proximity_relationships
from relationships.hanging import cal_hanging_relationships
from relationships.multi_objs import find_aligned_furniture, find_middle_furniture


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

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

def init_camera_view():
    camera_view = [0, -1, 0]
    camera_pos = [0, 0, 0]
    camera_view = camera_view / np.linalg.norm(camera_view)

    if camera_view[0] < 0:
        camera_angle = -utils.get_theta(camera_view, [0, 1, 0])
    else:
        camera_angle = utils.get_theta(camera_view, [0, 1, 0])

    return camera_view, camera_pos, camera_angle

def filter_bad_label(input_label):
    bad_label_list = ['ceiling', 'wall', 'door', 'doorframe', 'object']
    for o in bad_label_list:
        if o in input_label:
            return False

    return True

def get_obj_room_id (org_id):
    infos = org_id.split('|')
    if infos[1] == 'surface':
        return int(infos[2])
    else:
        return int(infos[1])

def generate_object_info(save_root, scene_name) :
    object_json_list = []

    inst2label_path = save_root / 'instance_id_to_label'
    pcd_path = save_root / 'pcd_with_global_alignment'

    inst_to_label = torch.load(inst2label_path / f"{scene_name}.pth")
    pcd_data = torch.load(pcd_path / f'{scene_name}.pth')

    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    pcds = np.concatenate([points, colors], 1)

    x_max, y_max, z_max = points.max(axis=0)
    x_min, y_min, z_min = points.min(axis=0)

    obj_pcds = []
    for i in np.unique(instance_labels):
        if i < 0:
            continue
        mask = instance_labels == i     # time consuming
        obj_pcds.append((pcds[mask], inst_to_label[int(i)], i))

    for _, (obj, obj_label, i) in enumerate(obj_pcds):
        gt_center, gt_size = convert_pc_to_box(obj)
        object_json = {
            'id': int(i),
            'label': obj_label,
            'position': gt_center,
            'size': gt_size,
            'mesh': None
        }
        object_json_list.append(object_json)

    # add scan_id
    object_json_string = {
        'scan': scene_name,
        'point_max': [x_max, y_max, z_max],
        'point_min': [x_min, y_min, z_min],
        'object_json_string': object_json_list,
        'inst_to_label': inst_to_label,
    }

    return object_json_string

def generate_ssg_data(dataset, scene_path, pre_load_path):
    ssg_data = {}
    pre_load_file_save_path = pre_load_path / (dataset + '.pkl')
    if pre_load_file_save_path.exists():
        print('Using preprocessed scene data')
        with open(pre_load_file_save_path, 'rb') as f:
            ssg_data = pickle.load(f)
    else:
        print('Preprocessing scene data')
        scans = [s.stem for s in (scene_path / 'pcd_with_global_alignment').glob('*.pth')]
        scans.sort()
        for scan_id in tqdm(scans):
            object_json_string = generate_object_info(scene_path, scan_id)
            if object_json_string is not None:
                ssg_data[scan_id] = object_json_string
        with open(pre_load_file_save_path, 'wb') as f:
            pickle.dump(ssg_data, f)

    return ssg_data

def main(cfg):
    cfg.rels_save_path.mkdir(parents=True, exist_ok=True)
    ssg_data = generate_ssg_data(cfg.dataset, cfg.scene_path, cfg.rels_save_path)
    scans_all = list(ssg_data.keys())

    ### init camera ###
    camera_view, camera_pos, camera_angle = init_camera_view()
    for scan_id in scans_all:
        objects_save = {}
        relationship_save = {}
        inst_dict = {}

        print('Processing ', scan_id)

        objects_info = ssg_data[scan_id]['object_json_string']
        inst_labels = ssg_data[scan_id]['inst_to_label']
        # bad case
        if len(objects_info) == 0:
            continue

        # construct object graph
        G = nx.DiGraph()
        # create nodes
        ObjNode_dict = {}

        # log objects of the same category
        for inst in inst_labels:
            if inst_labels[inst] not in inst_dict:
                inst_dict[inst_labels[inst]] = 1
            else:
                inst_dict[inst_labels[inst]] += 1

        x_max, y_max, z_max = ssg_data[scan_id]['point_max']
        x_min, y_min, z_min = ssg_data[scan_id]['point_min']
        scene_center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])
        # floor bad
        if z_max == z_min:
            z_max = z_min + 5
        scene_high = z_max - z_min

        # generate object node for graph
        obj_z_min = 1000
        floor_idx = -100
        for obj in objects_info:
            if np.array(obj['size']).sum() == 0:
                continue
            if not filter_bad_label(obj['label']):
                continue
            if obj['label'] == 'floor':
                floor_idx = int(obj['id'])
            node = ObjNode(id=int(obj['id']),
                           position=obj['position']-scene_center,
                           label=obj['label'],
                           mesh=obj['mesh'] if 'mesh' in obj else None,
                           size=np.array(obj['size']),
                           children=obj['children'] if 'children' in obj else None,
                           room_id=get_obj_room_id (obj['id_org']) if 'id_org' in obj else None,
                           dataset=cfg.dataset)

            if obj['position'][2] - obj['size'][2]/2 < obj_z_min:
                obj_z_min = obj['position'][2]-obj['size'][2]/2

            obj['count'] = inst_dict[node.label]
            obj['caption'] = ''

            ObjNode_dict[int(obj['id'])] = node
            G.add_node(node.id, label=node.label)

        # added special nodes (wall camera)
        G.add_node(-1, label='CAMERA')
        G.add_node(-2, label='wall')

        # special node for floor
        if floor_idx == -100:
            G.add_node(-3, label='floor')
            fx, fy, fz = scene_center[0], scene_center[1], obj_z_min
            node = ObjNode(id=-3,
                           position=np.array([fx, fy, fz]) - scene_center,
                           label='floor',
                           size=[(x_max-x_min)*1.2, (y_max-y_min)*1.2, (z_max-z_min)*0.1],
                           dataset=cfg.dataset)
            ObjNode_dict[-3] = node
            floor_idx = -3
        else:
            fx, fy, fz = scene_center[0], scene_center[1], obj_z_min
            node_ = ObjNode_dict[floor_idx]
            if node_.size[2] > 0:
                node = ObjNode(id=floor_idx,
                               position= np.array([fx, fy, fz]) - scene_center,
                               label='floor',
                               size=[max((x_max-x_min)*1.2, node_.size[0]),
                                     max((y_max-y_min)*1.2, node_.size[0]),
                                     node_.size[2]],
                               dataset=cfg.dataset)
            else:
                node = ObjNode(id=floor_idx,
                               position= np.array([fx, fy, fz]) - scene_center,
                               label='floor',
                               size=[max((x_max-x_min)*1.2, node_.size[0]),
                                     max((y_max-y_min)*1.2, node_.size[0]),
                                     (z_max-z_min)*0.1],
                               dataset=cfg.dataset)

            ObjNode_dict[floor_idx] = node

        # support embedded relationship
        if cfg.dataset.lower() in ['procthor']:
            support_relations = []
            embedded_relations = []
            hanging_objs_dict = {}
            for src_id, _ in ObjNode_dict.items():
                src_obj = ObjNode_dict[src_id]
                if src_obj.z_min <= ObjNode_dict[floor_idx].z_max and src_obj.id != floor_idx:
                    support_relations.append(utils.generate_relation(floor_idx, src_id,'support'))
                    hanging_objs_dict[src_id] = 1

                if src_obj.children != []:
                    for child in src_obj.children:
                        hanging_objs_dict[child] = 1
                        if child not in ObjNode_dict:
                            continue
                        if ObjNode_dict[child].z_max < src_obj.z_max:
                            embedded_relations.append(utils.generate_relation(src_id, child ,'inside_express'))
                        else:
                            support_relations.append(utils.generate_relation(src_id, child , 'support'))

        else:
            support_relations, embedded_relations, hanging_objs_dict = cal_support_relations(ObjNode_dict, camera_angle)
        for rela in support_relations:
            target_obj_id, obj_id, _ = rela
            G.add_edge(target_obj_id, obj_id, label='support') # optimizer

        # hanging relationships
        hanging_relationships = cal_hanging_relationships(ObjNode_dict, hanging_objs_dict, camera_angle,
                                                          scene_high, dataset=cfg.dataset)

        # iterate graph to cal saptial relationships
        proximity_relations = []
        for node in G:
            neighbor = dict(nx.bfs_successors(G, source=node, depth_limit=1))
            if len(neighbor[node]) > 1:
                neighbor_objs = neighbor[node]
                proximity = cal_proximity_relationships(neighbor_objs, camera_angle, ObjNode_dict, scene_high)
                proximity_relations += proximity

        # added some special relations and oppo support relationships
        oppo_support_relations = []
        objects_rels = support_relations + embedded_relations + hanging_relationships
        for idx, rels in enumerate(objects_rels):
            src, tgt, rela = rels
            if rela == 'support':
                oppo_support_relations.append(utils.generate_relation(src, tgt, 'oppo_support'))

            if src == -2 or tgt == -2:
                continue
            src_label = ObjNode_dict[src].label
            tgt_label = ObjNode_dict[tgt].label

            if src_label in dictionary.added_hanging and dictionary.added_hanging[src_label] == tgt_label:
                objects_rels[idx][2] = 'hanging'
            if tgt_label in dictionary.added_hanging and dictionary.added_hanging[tgt_label] == src_label:
                objects_rels[idx][2] = 'hanging'

        # multi objects
        multi_objs_relationships = []

        # added aligned relationship
        furniture_list = list(ObjNode_dict.keys())
        aligned_furniture = find_aligned_furniture(furniture_list, ObjNode_dict, 0.065)

        for _, aligned_furni in enumerate(aligned_furniture):
            multi_objs_relationships.append([aligned_furni, 'Aligned'])

        # added in the middle of relationship
        middle_relationships = find_middle_furniture(proximity_relations, ObjNode_dict)

        # output json
        relationships_json_string = {
            'scan': scan_id,
            'camera_view': camera_view,
            'camera_position': camera_pos,
            'relationships': objects_rels + proximity_relations + oppo_support_relations,
            'multi_objs_relationships': multi_objs_relationships + middle_relationships,
        }

        np.random.shuffle(objects_rels)
        # visualize scene
        if cfg.visualize:
            vis_dataset(ObjNode_dict=ObjNode_dict,
                        relation=proximity_relations,
                        scene_path=cfg.scene_path,
                        scan_id=scan_id,
                        scene_center=scene_center)


        relationship_save[scan_id] = relationships_json_string
        objects_save[scan_id] = {"objects_info": objects_info,
                                    "inst_to_label" : inst_labels}

        print ('==> DONE')
        print('SCENE ', scan_id)
        print('OBJECTS ', len(ObjNode_dict))

        scan_path = cfg.rels_save_path / scan_id

        scan_path.mkdir(parents=True, exist_ok=True)
        print('SAVE', scan_path)
        with (scan_path / 'relationships.json').open('w') as file:
            json.dump(relationship_save, file, default=default_dump)
        with (scan_path / 'objects.json').open('w') as file:
            json.dump(objects_save, file, default=default_dump)
        print ('=====================\n')


if __name__ == '__main__':
    cfg = OmegaConf.create({
        'dataset': 'dataset',
        'scene_path': '/path/to/dir',
        'rels_save_path': '/output/path/to/dir',
        'visualize': True,
        'num_workers': 1,
    })

    cfg.scene_path = Path(cfg.scene_path) / cfg.dataset / 'scan_data'
    cfg.rels_save_path = Path(cfg.rels_save_path) / cfg.dataset

    main(cfg)
